

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./img/Logo(S).png">
  <source media="(prefers-color-scheme: light)" srcset="./img/Logo Inverse(S).png">
  <img alt="Preferred.AI" src="./img/Logo(S).png">
</picture>

# Semantically Constitutive Entities in Knowledge Graphs

This repository contains the code for training Semantically Constitutive Entities (SemCon), as well as evaluation on the Link Prediction task.

## Usage

We recommend creating a [Conda](https://conda.io) environment named `semcon` via the provided [`environment.yml`](environment.yml):
```shell
conda env create -f environment.yml
conda activate semcon
```
## Learning Knowledge Graph Embeddings

We provide an example of using [OpenKE](https://github.com/thunlp/OpenKE) for Knowledge Graph (KG) embedding training. Please refer to the [CoDEx repository](https://github.com/tsafavi/codex#kge) for instructions on obtaining the CoDEx dataset.

We first clone the OpenKE repository and insert `TesterMaxRank`, which assigns the maximum rank for all tied items:
```shell
git clone https://github.com/thunlp/OpenKE.git
cd OpenKE && git checkout 4b8adac
cp -ai ../learn_kg_embs/openke_mod/* openke/
cp openke/base/Base.cpp openke/base/Base.original
awk 'NR==6{print "#include \"TestMaxRank.h\""}1' openke/base/Base.original > openke/base/Base.cpp
cd openke && bash make.sh
```

The following commands trains (TransE/TransH/TransD) KG embeddings on the FB15K237 dataset using [OpenKE-suggested hyperparameters](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/examples), and saves them in the given directory (`REPLACE_WITH_KG_DIR`):
```shell
# TransE
cd learn_kg_embs
python learn_kg_embs.py --in_path ../OpenKE/benchmarks/FB15K237/ --out_path REPLACE_WITH_KG_DIR --nbatches 100 --threads 4 --sampling_mode normal --bern_flag 1 --filter_flag 1 --neg_ent 25 --neg_rel 0 --model TransE --dim 200 --p_norm 1 --norm_flag True --loss_margin 5 --train_times 1000 --alpha 1.0 --opt_method sgd 
```

```shell
# TransH
cd learn_kg_embs
python learn_kg_embs.py --in_path ../OpenKE/benchmarks/FB15K237/ --out_path REPLACE_WITH_KG_DIR --nbatches 100 --threads 4 --sampling_mode normal --bern_flag 1 --filter_flag 1 --neg_ent 25 --neg_rel 0 --model TransH --dim 200 --p_norm 1 --norm_flag True --loss_margin 4 --train_times 1000 --alpha 0.5 --opt_method sgd
```

```shell
# TransD
cd learn_kg_embs
python learn_kg_embs.py --in_path ../OpenKE/benchmarks/FB15K237/ --out_path REPLACE_WITH_KG_DIR --nbatches 100 --threads 4 --sampling_mode normal --bern_flag 1 --filter_flag 1 --neg_ent 25 --neg_rel 0 --model TransD --dim_e 200 --dim_r 200 --p_norm 1 --norm_flag True --loss_margin 4 --train_times 1000 --alpha 1.0 --opt_method sgd
```

### Evaluating baseline performance
We first process the embeddings learnt in `REPLACE_WITH_KG_DIR/{DATASET}_{KG_MODEL}` (e.g., `FB15K237_TransE`) , and save the output file `{DATASET}_{KG_MODEL}-{MODEL}-{P}-{G}.cooX` in the given directory (`REPLACE_WITH_OUTPUT_DIR`):
```shell
cd baselines
python -u train_kg_centrality_baselines.py --c_ij REPLACE_WITH_KG_DIR/{DATASET}_{KG_MODEL} --output_dir REPLACE_WITH_OUTPUT_DIR --degree_type all
```
Next, we run the evaluation script:
```shell
cd eval
python kg_link_pred.py --c_ij REPLACE_WITH_KG_DIR/{DATASET}_{KG_MODEL}/ --eval_X REPLACE_WITH_OUTPUT_DIR/{DATASET}_{KG_MODEL}-{MODEL}-{P}-{G}.cooX
```

## Training Semantically Constitutive Entities

The following commands trains Semantically Constitutive Entities via a two-step procedure, and saves them in the given directory (`REPLACE_WITH_DIR`):

Note: Requires a [Gurobi](https://www.gurobi.com/) license to run.

```shell
cd semantically_constitutive
# compile programs
make
# Step 1: Create relaxed solution at REPLACE_WITH_DIR/RELAXED_COOX_FILE
./relaxed_partial -c REPLACE_WITH_KG_DIR/{DATASET}_{KG_MODEL} -k 4000 -g 10 -o REPLACE_WITH_OUTPUT_DIR
# Step 2: Obtain integer solution, with REPLACE_WITH_DIR/RELAXED_COOX_FILE as initial solution
./relaxed_solver -c REPLACE_WITH_KG_DIR/{DATASET}_{KG_MODEL} -x REPLACE_WITH_OUTPUT_DIR/COOX_FILE -k 4000 -g 10 -o REPLACE_WITH_OUTPUT_DIR
```

Evaluation on the Link Prediction task can be done as above:
```shell
cd eval
python kg_link_pred.py --c_ij REPLACE_WITH_KG_DIR/{DATASET}_{KG_MODEL}/ --eval_X REPLACE_WITH_OUTPUT_DIR/COOX_FILE
```

-----
## Baselines

We provide example code for replicating the baseline models used in our paper. All output file are evaluated in the same manner as above.

### KG Centrality Measures
```shell
cd baselines
python train_kg_centrality_baselines.py --c_ij REPLACE_WITH_KG_DIR/{DATASET}_{KG_MODEL} --output_dir REPLACE_WITH_OUTPUT_DIR --degree_type group_in --p 4000 --g 10
```

### SSA
We use the SSA implementation from [@hungnt55](https://github.com/hungnt55/Stop-and-Stare).

```shell
cd baselines
git clone https://github.com/hungnt55/Stop-and-Stare.git
unzip Stop-and-Stare/SSA_release_2.1.zip
cp coox2bin.cpp SSA_release_2.0/SSA
cd SSA_release_2.0/SSA
echo -e "\tg++ coox2bin.cpp -o coox2bin \$(PARA)" >> Makefile
make
coox2bin REPLACE_WITH_KG_DIR/{DATASET}_{KG_MODEL} REPLACE_WITH_OUTPUT_DIR/{DATASET}_{KG_MODEL}-{MODEL}.bin
./SSA -i REPLACE_WITH_OUTPUT_DIR/{DATASET}_{KG_MODEL}.bin -k 4000 -epsilon 0.03 -delta 0.01 -m LT > {DATASET}_{KG_MODEL}_SSA.out
python process_ssa_output.py --input {DATASET}_{KG_MODEL}_SSA.out --g 10 --output_dir REPLACE_WITH_OUTPUT_DIR
```

### PageRank

We use the PageRank implementation from [@louridas](https://github.com/louridas/pagerank).
```shell
cd baselines
git clone https://github.com/louridas/pagerank.git
cd pagerank/cpp && make
./pagerank -n -a 0.85 -c 0.00000000000000000000000000000001 REPLACE_WITH_KG_DIR/{DATASET}_{KG_MODEL} > REPLACE_WITH_OUTPUT_DIR/{DATASET}_{KG_MODEL}.pr
python process_pagerank_output.py --c_ij REPLACE_WITH_KG_DIR/{DATASET}_{KG_MODEL} --pr_out REPLACE_WITH_OUTPUT_DIR/{DATASET}_{KG_MODEL}.pr --p 4000 --g 10 --output_dir REPLACE_WITH_OUTPUT_DIR
```

## Citiation
Our paper can be cited in the following formats:

### APA
```text
Chia, C., Tkachenko, M., & Lauw, H. (2022). Semantically Constitutive Entities in Knowledge Graphs. In Database and Expert Systems Applications: 34th International Conference, DEXA 2023.
```

### Bibtex
```bibtex
@inproceedings{chia2023semantically,
    title={Semantically Constitutive Entities in Knowledge Graphs},
    author={Chia, Chong Cher and Tkachenko, Maksim and Lauw, Hady W},
    booktitle={Database and Expert Systems Applications: 34th International Conference, DEXA 2023, Penang, Malaysia, August 28-30, 2023, Proceedings},
    year={2023},
    publisher={Springer Nature}
}
```