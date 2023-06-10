import itertools

from collections import defaultdict

import argparse
import logging
import numpy as np

from pathlib import Path

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_id_file(input_f: Path):
    with open(input_f, 'r') as f:
        num_lines = int(f.readline().strip())
        result = [tuple(int(i) for i in line.strip().replace(' ', '\t').split('\t')) for line in f]
        assert len(result) == num_lines
    return result


def main(args):

    pagerank_scores = {int(i): float(v) for i, v in
                       [line.split(' = ') for line in itertools.islice(open(args.pr_out, 'r'), 1, None)]}
    facilities = [*sorted(pagerank_scores.keys(), key=lambda x: pagerank_scores[x], reverse=True)[:args.p]]
    num_ents = len(np.loadtxt(args.c_ij / 'entembs.npy'))

    # NOTE: this only uses true triplets from KG for degree counting
    training_triples = load_id_file(args.c_ij / 'train2id.txt')
    validation_triples = load_id_file(args.c_ij / 'valid2id.txt')
    true_triples = training_triples + validation_triples

    valid_facilities = defaultdict(set)
    for (h, t, r) in true_triples:
        valid_facilities[t].add(h)

    x_coo = []
    for i in range(num_ents):
        if i in facilities:
            x_coo.append((i, i, 1))
        else:
            valid_js = valid_facilities[i]
            recon_js = [j for j in facilities if j in valid_js][:args.g]
            x_coo.extend([(i, j, 1/len(recon_js)) for j in recon_js])

    out_filename = f'{args.c_ij.name}-pr_{args.pr_out.stem.split("_")[-1]}-{args.p}-{args.g}'
    with open(args.output_dir / f'{out_filename}.cooX', 'w') as f:
        f.write(f'{num_ents}\t{num_ents}\n')
        f.write('\n'.join([f'{i}\t{j}\t{x_ij}' for i, j, x_ij in x_coo]))

    logger.info(f'DONE: {args.output_dir}/{out_filename}.cooX')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--c_ij', type=Path)
    argparser.add_argument('--pr_out', type=Path)
    argparser.add_argument('--output_dir', type=Path)
    argparser.add_argument('--p', type=int)
    argparser.add_argument('--g', type=int)
    args = argparser.parse_args()

    assert args.c_ij.is_dir(), f'could not find c_ij dir: {args.c_ij}'

    main(args)
