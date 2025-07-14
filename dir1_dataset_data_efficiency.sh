#!/bin/bash

# # Cora ML
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_dir.py --dataset cora_ml --split_ratio $ratio --type txt
#     wait
# done

# Cora
for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python split_datasets_dir.py --dataset cora --split_ratio $ratio --type pt
    wait
done

for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python split_datasets_dir.py --dataset cora --split_ratio $ratio --type txt
    wait
done

# # CiteSeer
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_dir.py --dataset citeseer --split_ratio $ratio --type txt
#     wait
# done

# # PubMed
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_dir.py --dataset pubmed --split_ratio $ratio --type txt
#     wait
# done