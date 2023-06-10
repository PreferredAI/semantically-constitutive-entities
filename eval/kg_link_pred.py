import argparse
import logging

from collections import defaultdict
from itertools import islice
from pathlib import Path
from timeit import default_timer as timer

import numpy as np

from scipy.sparse import csr_array
from scipy.stats import rankdata
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


def link_pred_filtered(testing_triples, entity_set, true_heads, true_tails, ent_embs, rel_embs, removed_ents):
    ent_embs = ent_embs / np.linalg.norm(ent_embs, 2, -1, keepdims=True)
    rel_embs = rel_embs / np.linalg.norm(rel_embs, 2, -1, keepdims=True)

    filtered_head_ranks, filtered_tail_ranks = [], []
    for (test_head, test_tail, test_rel) in testing_triples:
        if test_head in removed_ents or test_tail in removed_ents:
            filtered_head_ranks.append(len(entity_set))
            filtered_tail_ranks.append(len(entity_set))
            continue

        filtered_heads = [test_head, *(entity_set - true_heads[(test_rel, test_tail)])]
        filtered_tails = [test_tail, *(entity_set - true_tails[(test_head, test_rel)])]

        _fixed_head_emb = ent_embs[test_tail] - rel_embs[test_rel]
        filtered_head_scores = np.sum(np.abs(_fixed_head_emb[np.newaxis, :] - ent_embs[filtered_heads, :]), axis=1)
        filtered_head_rank = rankdata(filtered_head_scores, method='max', nan_policy='omit')[0]
        filtered_head_ranks.append(filtered_head_rank)

        _fixed_tail_emb = ent_embs[test_head] + rel_embs[test_rel]
        filtered_tail_scores = np.sum(np.abs(ent_embs[filtered_tails, :] - _fixed_tail_emb[np.newaxis, :]), axis=1)
        filtered_tail_rank = rankdata(filtered_tail_scores, method='max', nan_policy='omit')[0]
        filtered_tail_ranks.append(filtered_tail_rank)

    return filtered_head_ranks, filtered_tail_ranks


def main(args):
    global ent_embeddings, rel_embeddings, rel_matrix, real_reconstructions

    ent_embeddings = np.loadtxt(args.c_ij / 'entembs.npy')
    rel_embeddings = np.loadtxt(args.c_ij / 'relembs.npy')

    _filtered_ents = np.loadtxt(args.c_ij / 'filtered_ents.npy', dtype=int)
    num_ents = _filtered_ents[0]
    entity_set = {i for i in _filtered_ents[1:]}

    coo_rels = np.array([int(i) for i in islice(open(args.c_ij / 'rels.tsv'), 1, None)])
    coo_costs = np.array([float(i.strip()) for i in islice(open(args.c_ij / 'costs.tsv'), 1, None)])
    coo_rows = np.array([int(i) for i in islice(open(args.c_ij / 'rows.tsv'), 1, None)])
    coo_cols = np.array([int(i) for i in islice(open(args.c_ij / 'cols.tsv'), 1, None)])
    rel_matrix = csr_array((coo_rels, (coo_rows, coo_cols)), shape=(num_ents, num_ents))
    costs_matrix = csr_array((coo_costs, (coo_rows, coo_cols)), shape=(num_ents, num_ents))
    valid_locations = defaultdict(set)
    for i, j in zip(coo_rows, coo_cols):
        valid_locations[j].add(i)

    logger.info(f'loading triples from {args.c_ij}')
    training_triples = load_id_file(args.c_ij / 'train2id.txt')
    validation_triples = load_id_file(args.c_ij / 'valid2id.txt')
    testing_triples = load_id_file(args.c_ij / 'test2id.txt')

    # Using all triples, to replicate OpenKE results
    true_triples = training_triples + validation_triples + testing_triples
    true_heads, true_tails = defaultdict(set), defaultdict(set)
    real_reconstructions = {i: set() for i in range(len(ent_embeddings))}
    for (h, t, r) in true_triples:
        real_reconstructions[h].add(t)  # h (facility j) + r => t (location i)
        true_heads[(r, t)].add(h)
        true_tails[(h, r)].add(t)

    reconed_embs = np.empty_like(ent_embeddings)
    failed_i = set()
    coo_costs = 0
    recon_stats = {'direct': [],
                   'negrel': [],
                   'failed': [],
                   'memorized': [],
                   'self': [],
                   'num_recon_facilities': []}

    with open(args.eval_X, 'r') as f:
        arr_shape = tuple(int(i) for i in f.readline().strip().split('\t'))
        coo_rows, coo_cols, coo_vals = [], [], []
        for line in f:
            record = line.strip().split('\t')
            _row, _col, _val = int(record[0]), int(record[1]), float(record[2])
            if _val > 0.0001:
                coo_rows.append(_row)
                coo_cols.append(_col)
                coo_vals.append(_val)
        x_hat = csr_array((coo_vals, (coo_rows, coo_cols)), shape=arr_shape)

    y_hat = [1 if i > 0 else 0 for idx, i in enumerate(np.sum(x_hat, axis=0))]
    if any([d > len(ent_embeddings) for d in x_hat.shape]):
        # for ghost facilities, assign weights back to real reconstructions
        num_ghost_facilities = 2 * int(args.eval_X.stem.split('-')[-1])
        logger.info(f'num ghost facilities detected: {num_ghost_facilities // 2}')
        real_location_assignments = x_hat[:, :-num_ghost_facilities]
        scale_factor = num_ghost_facilities / np.sum(real_location_assignments, axis=1)

        x_hat = csr_array((real_location_assignments * scale_factor[:, np.newaxis]) / num_ghost_facilities)
        y_hat = y_hat[:-num_ghost_facilities]

    facilities = {i for i, v in enumerate(y_hat) if v}
    logger.info('reconstructing embeddings..')
    for i, assignment in tqdm(enumerate(x_hat), total=len(x_hat)):
        if i in facilities:  # memorized, no need to calculate reconstruction
            reconed_embs[i] = ent_embeddings[i]
            recon_stats['self'].append(i)
            recon_stats['num_recon_facilities'].append(0)
            continue

        facility_weights = {j: assignment[0, j] for j in assignment.indices if assignment[0, j] > 0}
        if not facility_weights:
            recon_stats['failed'].append(i)
            failed_i.add(i)
            continue

        reconstruction = []
        sum_w = sum(facility_weights.values())
        recon_stats['num_recon_facilities'].append(len(facility_weights))

        for j, w in facility_weights.items():
            rel_idx = rel_matrix[i, j]
            if rel_idx < -1:  # convert from: -2 * (min_neg_rel_idx + 1)
                rel_idx = int((rel_idx / -2) - 1)
                recon_stats['negrel'].append(i)
                rel_emb = rel_embeddings[rel_idx]
                recon_emb = ent_embeddings[j] - rel_emb
            else:
                if rel_idx == -1:
                    recon_stats['memorized'].append(i)
                    rel_emb = np.zeros_like(ent_embeddings[j])
                else:
                    recon_stats['direct'].append(i)
                    rel_emb = rel_embeddings[rel_idx]
                recon_emb = ent_embeddings[j] + rel_emb
            reconstruction.append((w / sum_w) * recon_emb)

            coo_costs += costs_matrix[i, j] if i in valid_locations[j] else 1E10

        reconstruction = np.sum(reconstruction, axis=0)
        reconed_embs[i] = reconstruction

    start_time = timer()
    full_filtered_head_ranks, full_filtered_tail_ranks = link_pred_filtered(testing_triples,
                                                                            entity_set,
                                                                            true_heads,
                                                                            true_tails,
                                                                            reconed_embs,
                                                                            rel_embeddings,
                                                                            removed_ents=failed_i)
    end_time = timer()
    logger.info(f'reconstruction done in {end_time - start_time:0.2f} seconds')

    # calculate stats
    ranks = np.array(full_filtered_head_ranks + full_filtered_tail_ranks)
    hit_10_n = int(len(ent_embeddings) * 0.1)
    hit_10p = sum(ranks <= hit_10_n) / len(ranks)
    hit_05_n = int(len(ent_embeddings) * 0.05)
    hit_05p = sum(ranks <= hit_05_n) / len(ranks)
    hit_01_n = int(len(ent_embeddings) * 0.01)
    hit_01p = sum(ranks <= hit_01_n) / len(ranks)
    mrr = np.average([1/i for i in ranks])

    logger.info(f'{args.eval_X.stem}\t{len(facilities)}\t'
                f'num_testing_triples\t{len(testing_triples)}\t'
                f'mrr\t{mrr:0.6f}\t'
                f'hit_10%\t{hit_10p:0.6f}\t'
                f'hit_5%\t{hit_05p:0.6f}\t'
                f'hit_1%\t{hit_01p:0.6f}\t')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--c_ij', type=Path)
    argparser.add_argument('--eval_X', type=Path)

    args = argparser.parse_args()
    assert args.c_ij.is_dir(), f'could not find c_ij dir: {args.c_ij}'

    main(args)
