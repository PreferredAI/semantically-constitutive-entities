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

    num_ents = len(np.loadtxt(args.c_ij / 'entembs.npy'))

    # NOTE: this only uses true triplets from KG for degree counting
    training_triples = load_id_file(args.c_ij / 'train2id.txt')
    validation_triples = load_id_file(args.c_ij / 'valid2id.txt')

    graph = defaultdict(dict)
    true_triples = training_triples + validation_triples
    valid_facilities = defaultdict(set)
    for (h, t, r) in true_triples:
        graph[h][t] = 1
        valid_facilities[t].add(h)

    with open(args.input, 'r') as f:
        lines = f.read().split('\n')
        seed_nodes = [i for i in lines if i[:11] == 'Seed Nodes:']
        time = [i for i in lines if i[:5] == 'Time:']
        assert len(seed_nodes) == 1, f'{len(seed_nodes)}: {[f"{i[:20]}" for i in seed_nodes]}'
        assert len(time) == 1, f'{len(time)}: {[f"{i[:20]}" for i in time]}'

    facilities = [int(i) for i in seed_nodes[0][11:].strip().split()]  # in (D)SSA order
    assert len(facilities) == args.p
    out_filename = f'{args.c_ij.name}-ssa-{args.p}-{args.g}'
    x_coo = []
    for i in range(num_ents):
        if i in facilities:
            x_coo.append((i, i, 1))
        else:
            valid_js = valid_facilities[i]
            recon_js = [j for j in facilities if j in valid_js][:args.g]
            x_coo.extend([(i, j, 1/len(recon_js)) for j in recon_js])
    with open(args.output_dir / (out_filename + '.cooX'), 'w') as f:
        f.write(f'{num_ents}\t{num_ents}\n')
        f.write('\n'.join([f'{i}\t{j}\t{x_ij}' for i, j, x_ij in x_coo]))

    logger.info(f'\n\ndone in {time[0][5:-1]} seconds\t{out_filename}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=Path)
    argparser.add_argument('--c_ij', type=Path)
    argparser.add_argument('--output_dir', type=Path)
    argparser.add_argument('--p', type=int)
    argparser.add_argument('--g', type=int)
    args = argparser.parse_args()

    main(args)
