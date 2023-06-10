import argparse
import itertools
import logging
import sys

from collections import defaultdict
from pathlib import Path

sys.path.append("..")  # for src/OpenKE
from OpenKE.openke.config import Trainer
from OpenKE.openke.config.TesterMaxRank import Tester
from OpenKE.openke.data import TestDataLoader
from OpenKE.openke.data import TrainDataLoader
from OpenKE.openke.module.loss import MarginLoss
from OpenKE.openke.module.model import TransD
from OpenKE.openke.module.model import TransE
from OpenKE.openke.module.model import TransH
from OpenKE.openke.module.strategy import NegativeSampling

import numpy as np
import torch


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

models = {
    'TransE': TransE,
    'TransH': TransH,
    'TransD': TransD,
}
model_kwargs = {  # all models have ent_tot, rel_tot
    'TransE': {'dim', 'p_norm', 'norm_flag', 'margin', 'epsilon'},
    'TransH': {'dim', 'p_norm', 'norm_flag', 'margin', 'epsilon'},
    'TransD': {'dim_e', 'dim_r', 'p_norm', 'norm_flag', 'margin', 'epsilon'},
}


def main(args):
    out_dir = Path(f'{args.out_path}/{args.in_path.name}_{args.model}')
    # out_filename = f'{args.model}_{args.in_path.name}'
    out_dir.mkdir(exist_ok=True, parents=True)

    run_model_kwargs = {k: v for k, v in vars(args).items() if k in model_kwargs[args.model]}    

    train_dataloader = TrainDataLoader(
        in_path=f'{args.in_path.resolve()}/',
        nbatches=args.nbatches,
        batch_size=args.batch_size,
        threads=args.threads,
        sampling_mode=args.sampling_mode,
        bern_flag=args.bern_flag,
        filter_flag=args.filter_flag,
        neg_ent=args.neg_ent,
        neg_rel=args.neg_rel)

    # dataloader for test
    test_dataloader = TestDataLoader(f'{args.in_path.resolve()}/', "link")

    # define the model
    model = models[args.model](ent_tot=train_dataloader.get_ent_tot(),
                               rel_tot=train_dataloader.get_rel_tot(),
                               **run_model_kwargs)

    # define the loss function
    model_loss = NegativeSampling(model=model,
                                  loss=MarginLoss(args.loss_margin),
                                  batch_size=train_dataloader.get_batch_size())

    # train the model
    trainer = Trainer(model=model_loss, data_loader=train_dataloader, train_times=args.train_times, alpha=args.alpha, use_gpu=True)
    trainer.run()
    model.save_checkpoint(out_dir / f'{args.in_path.name}_{args.model}.ckpt')

    # test the model
    logger.info(f'OpenKE Link Prediction (Max Ranking)..')
    model.load_checkpoint(out_dir / f'{args.in_path.name}_{args.model}.ckpt')
    tester = Tester(model=model, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction_maxrank(type_constrain=False)

    # make filtered experiment datasets
    state_dict = torch.load(out_dir / f'{args.in_path.name}_{args.model}.ckpt', map_location=args.device)
    ent_embeddings = state_dict['ent_embeddings.weight'].cpu().detach().numpy()
    rel_embeddings = state_dict['rel_embeddings.weight'].cpu().detach().numpy()

    logger.warning(f'only using train&&val reconstructions from: {args.in_path.resolve()}')

    train_triples = [tuple(int(i) for i in line.strip().split())
                     for line in itertools.islice(open(args.in_path / 'train2id.txt'), 1, None)]
    valid_triples = [tuple(int(i) for i in line.strip().split())
                     for line in itertools.islice(open(args.in_path / 'valid2id.txt'), 1, None)]

    # only used for generating filtered dataset
    test_triples = [tuple(int(i) for i in line.strip().split())
                    for line in itertools.islice(open(args.in_path / 'test2id.txt'), 1, None)]

    training_triples = train_triples + valid_triples
    recon_facilities = defaultdict(set)
    for (h, t, r) in training_triples:  # (h, t, r); h + r => t
        recon_facilities[t].add(h)

    actual_entembs_path = f'{out_dir}/entembs.npy'
    np.savetxt(actual_entembs_path, ent_embeddings)
    actual_relembs_path = f'{out_dir}/relembs.npy'
    np.savetxt(actual_relembs_path, rel_embeddings)

    filter_num_edges = 0
    costs_lookup, rel_lookup = dict(), dict()
    valid_entities = {ent for ent, facilities in recon_facilities.items() if len(facilities) >= filter_num_edges}

    for (head_ent, tail_ent, rel_idx) in training_triples:
        if tail_ent not in valid_entities:  # reject triplets with locations (tail ent) that cannot be reconstructed
            continue

        # h (facility j) + r => t (location i);
        recon_cost = np.sum((ent_embeddings[head_ent] + rel_embeddings[rel_idx] - ent_embeddings[tail_ent])**2)

        # negative relations; allow reversed reconstruction
        neg_rel_cost = costs_lookup.get((head_ent, tail_ent))
        if neg_rel_cost is None:
            costs_lookup[(tail_ent, head_ent)] = recon_cost
            rel_lookup[(tail_ent, head_ent)] = rel_idx
        elif recon_cost < neg_rel_cost:  # t (facility j) - r => h (location i);
            costs_lookup[(tail_ent, head_ent)] = recon_cost
            rel_lookup[(tail_ent, head_ent)] = rel_idx
            costs_lookup[(head_ent, tail_ent)] = recon_cost
            rel_lookup[(head_ent, tail_ent)] = -2 * (rel_idx + 1)
        elif recon_cost >= neg_rel_cost:  # use neg_rel instead
            costs_lookup[(tail_ent, head_ent)] = neg_rel_cost
            neg_rel_rel_idx = rel_lookup[(head_ent, tail_ent)]
            if neg_rel_rel_idx >= 0:
                rel_lookup[(tail_ent, head_ent)] = -2 * (neg_rel_rel_idx + 1)
            else:
                rel_lookup[(tail_ent, head_ent)] = int((neg_rel_rel_idx / -2) - 1)

    # self-recon
    for ent_idx in valid_entities:
        costs_lookup[(ent_idx, ent_idx)] = 0
        rel_lookup[(ent_idx, ent_idx)] = -1

    coo_header = f'{len(costs_lookup)}'
    rows, cols, costs, rels = [coo_header], [coo_header], [coo_header], [coo_header]
    for (location_ent, facility_ent), current_cost in sorted(costs_lookup.items()):
        rows.append(f'{location_ent}')
        cols.append(f'{facility_ent}')
        costs.append(f'{current_cost}')
        rels.append(f'{rel_lookup[(location_ent, facility_ent)]}')

    with open(out_dir / f'rows.tsv', 'w') as f_rows, \
            open(out_dir / f'cols.tsv', 'w') as f_cols, \
            open(out_dir / f'costs.tsv', 'w') as f_costs, \
            open(out_dir / f'rels.tsv', 'w') as f_rels:

        f_rows.write('\n'.join(rows))
        f_cols.write('\n'.join(cols))
        f_costs.write('\n'.join(costs))
        f_rels.write('\n'.join(rels))

    np.savetxt(f'{out_dir}/filtered_ents.npy',
               np.array([len(ent_embeddings)] + sorted(valid_entities)),
               fmt='%s')

    for split, split_triples in zip(['train', 'valid', 'test'], [train_triples, valid_triples, test_triples]):
        filtered_split_triples = [f'{h} {t} {r}' for (h, t, r) in split_triples if t in valid_entities]
        with open(out_dir / f'{split}2id.txt', 'w') as f:
            f.write(f'{len(filtered_split_triples)}\n')
            f.write('\n'.join(filtered_split_triples))

    logger.info(f'saved to: {Path(actual_entembs_path).parent}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--in_path', type=Path)
    argparser.add_argument('--out_path', type=str)
    argparser.add_argument('--sampling_mode', type=str, choices=['normal', 'cross'])
    argparser.add_argument('--nbatches', type=int)
    argparser.add_argument('--batch_size', type=int)
    argparser.add_argument('--threads', type=int)
    argparser.add_argument('--bern_flag', type=int, choices=[0, 1])
    argparser.add_argument('--filter_flag', type=int, choices=[0, 1])
    argparser.add_argument('--neg_ent', type=int)
    argparser.add_argument('--neg_rel', type=int)
    argparser.add_argument('--loss_margin', type=float)
    argparser.add_argument('--train_times', type=int)
    argparser.add_argument('--alpha', type=float)
    argparser.add_argument('--opt_method', type=str)

    argparser.add_argument('--model', type=str, choices=['TransE', 'TransH', 'TransD'])
    argparser.add_argument('--dim', type=int)
    argparser.add_argument('--dim_e', type=int)
    argparser.add_argument('--dim_r', type=int)
    argparser.add_argument('--p_norm', type=int)
    argparser.add_argument('--norm_flag', type=bool)
    argparser.add_argument('--margin')
    argparser.add_argument('--epsilon')

    args = argparser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args)
