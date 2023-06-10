import argparse
import logging

from collections import Counter
from collections import defaultdict
from pathlib import Path
from random import Random
from timeit import default_timer as timer

import numpy as np

from tqdm import tqdm

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
    ent_embeddings = np.loadtxt(args.c_ij / 'entembs.npy')

    # NOTE: this only uses true triplets from KG for degree counting
    training_triples = load_id_file(args.c_ij / 'train2id.txt')
    validation_triples = load_id_file(args.c_ij / 'valid2id.txt')

    true_triples = training_triples + validation_triples
    true_heads, true_tails, valid_facilities, valid_locations = [defaultdict(set)] * 4
    real_reconstructions = {i: set() for i in range(len(ent_embeddings))}
    for (h, t, r) in true_triples:
        real_reconstructions[h].add(t)  # h (facility j) + r => t (location i)
        true_heads[(r, t)].add(h)
        true_tails[(h, r)].add(t)
        valid_facilities[t].add(h)
        valid_locations[h].add(t)

    heads, tails, _ = zip(*true_triples)
    num_ind_counter = Counter(tails)
    num_outd_counter = Counter(heads)

    timer_start = timer()
    x_coo = []
    if args.degree_type == 'all':
        '''
        Use original embeddings
        '''
        x_coo = [(i, i, 1) for i in range(len(ent_embeddings))]
    elif args.degree_type == 'random':
        seeded_random = Random(42)
        facilities = {*seeded_random.sample(range(len(ent_embeddings)), args.p)}
        for i in range(len(ent_embeddings)):
            if i in facilities:
                x_coo.append((i, i, 1))
            else:
                rand_js = sorted({*(valid_facilities[i])} & facilities)
                if len(rand_js) > args.g:
                    rand_js = seeded_random.sample(rand_js, args.g)
                x_coo.extend([(i, j, 1/len(rand_js)) for j in rand_js])
    elif args.degree_type[:5] == 'point':
        '''
        Only calculate edges once, then select up to P facilities for reconstruction by order
        '''
        if args.degree_type == 'point_in':  # select p facilities that have the most in-degrees
            sorted_degree_facilities = num_ind_counter.most_common(args.p)
        elif args.degree_type == 'point_out':  # select p facilities that have the most out-degrees
            sorted_degree_facilities = num_outd_counter.most_common(args.p)

        facilities, _ = zip(*sorted_degree_facilities)
        for i in range(len(ent_embeddings)):
            if i in facilities:
                x_coo.append((i, i, 1))
            else:
                valid_js = valid_facilities[i]
                recon_js = [j for j in facilities if j in valid_js][:args.g]
                x_coo.extend([(i, j, 1/len(recon_js)) for j in recon_js])
    elif args.degree_type[:5] == 'group':
        '''
        Recalculate edges after each step, then select up to P facilities for reconstruction by order
        '''

        sorted_degree_facilities = []
        _fs = set()
        triple_cache = [(h, t) for (h, t, _) in true_triples]
        tqdm_bar = tqdm(total=args.p)

        while len(sorted_degree_facilities) < args.p:
            if args.degree_type == 'group_in':
                sorted_degree_facilities.append(num_ind_counter.most_common(1)[0])
            elif args.degree_type == 'group_out':
                sorted_degree_facilities.append(num_outd_counter.most_common(1)[0])
            else:
                assert False, f'invalid degree greedy type: {args.degree_type}'
            _fs.add(sorted_degree_facilities[-1][0])
            # count heads/tails for triples that do not involve already selected facilities
            try:
                coo_cache = [(h, t) for (h, t) in triple_cache if not ({h, t} & _fs)]
                heads, tails = zip(*coo_cache)
            except ValueError:  # no more triples; done with selection!
                break
            num_ind_counter = Counter(tails)    # h + r => tail
            num_outd_counter = Counter(heads)  # h + r => tail
            tqdm_bar.update(1)

        facilities, degrees = zip(*sorted_degree_facilities)

        # if insufficient facilities, randomly select from remaining locations
        if len(facilities) < args.p:
            choices = sorted({*range(len(ent_embeddings))} - facilities)
            seeded_rand = Random(42)
            new_facilities = {*seeded_rand.sample(choices, args.p - len(facilities))}
            facilities = facilities | new_facilities

        for i in range(len(ent_embeddings)):
            if i in facilities:
                x_coo.append((i, i, 1))
            else:
                valid_js = valid_facilities[i]
                recon_js = [j for j in facilities if j in valid_js][:args.g]
                x_coo.extend([(i, j, 1/len(recon_js)) for j in recon_js])
    else:
        assert False, f'unsupported degree type: {args.degree_type}'

    timer_end = timer()

    out_filename = f'{args.c_ij.name}-{args.degree_type}-{args.p}-{args.g}'
    args.output_dir.mkdir(exist_ok=True, parents=True)
    with open(args.output_dir / f'{out_filename}.cooX', 'w') as f:
        f.write(f'{len(ent_embeddings)}\t{len(ent_embeddings)}\n')
        f.write('\n'.join([f'{i}\t{j}\t{x_ij}' for i, j, x_ij in x_coo]))

    logger.info(f'\n\ndone in {timer_end - timer_start:.2f} seconds\t{args.output_dir/out_filename}')


if __name__ == '__main__':

    degree_types = ['point_in', 'point_out', 'group_in', 'group_out', 'random', 'all']

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--c_ij', type=Path)
    argparser.add_argument('--output_dir', type=Path)
    argparser.add_argument('--degree_type', choices=degree_types)
    argparser.add_argument('--p', type=int)
    argparser.add_argument('--g', type=int)
    args = argparser.parse_args()

    main(args)
