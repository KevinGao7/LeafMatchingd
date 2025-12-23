# #!/bin/bash

# # Cora ML
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_dir2.py --dataset cora_ml --split_ratio $ratio --type pt &
#     python split_datasets_dir2.py --dataset cora_ml --split_ratio $ratio --type txt &
#     wait
# done

# # Cora
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_dir2.py --dataset cora --split_ratio $ratio --type pt &
#     python split_datasets_dir2.py --dataset cora --split_ratio $ratio --type txt &
#     wait
# done

# # CiteSeer
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_dir2.py --dataset citeseer --split_ratio $ratio --type pt &
#     python split_datasets_dir2.py --dataset citeseer --split_ratio $ratio --type txt &
#     wait
# done

# # PubMed
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_dir2.py --dataset pubmed --split_ratio $ratio --type pt &
#     python split_datasets_dir2.py --dataset pubmed --split_ratio $ratio --type txt &
#     wait
# done

# Amazon
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_dir2.py --dataset amazon --split_ratio $ratio --type pt &
#     python split_datasets_dir2.py --dataset amazon --split_ratio $ratio --type txt &
# done
# wait

# # fly-larva
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_dir2.py --dataset fly_larva --split_ratio $ratio --type pt &
#     python split_datasets_dir2.py --dataset fly_larva --split_ratio $ratio --type txt &
# done
# wait

# # yeast
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_dir2.py --dataset yeast --split_ratio $ratio --type pt &
#     python split_datasets_dir2.py --dataset yeast --split_ratio $ratio --type txt &
# done

# # male
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_dir2.py --dataset male --split_ratio $ratio --type pt &
#     python split_datasets_dir2.py --dataset male --split_ratio $ratio --type txt &
# done


# ICEWS18
for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python split_datasets_dir2.py --dataset icews18_min --split_ratio $ratio --type pt &
    python split_datasets_dir2.py --dataset icews18_min --split_ratio $ratio --type txt &
    python split_datasets_dir2.py --dataset icews18_mid --split_ratio $ratio --type pt &
    python split_datasets_dir2.py --dataset icews18_mid --split_ratio $ratio --type txt &
    python split_datasets_dir2.py --dataset icews18_max --split_ratio $ratio --type pt &
    python split_datasets_dir2.py --dataset icews18_max --split_ratio $ratio --type txt &    
    wait
done

# # twitter
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_dir2.py --dataset twitter --split_ratio $ratio --type pt &
#     python split_datasets_dir2.py --dataset twitter --split_ratio $ratio --type txt &
#     wait
# done

# # twitter-scaling
# for ratio in 0.0078125 0.00390625 0.001953125 0.0009765625; do
#     python split_datasets_dir2.py --dataset twitter --split_ratio $ratio --type pt &
# done

# # citation-scaling
# for ratio in 0.25 0.125 0.0625 0.03125 0.015625; do
#     python split_datasets_dir2.py --dataset ogbl_citation2 --split_ratio $ratio --type pt &
# done

# ogbl-citation2
# for ratio in 0.02 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# do
#     # python split_datasets_dir2.py --dataset ogbl_citation2 --split_ratio $ratio --type pt &
#     python split_datasets_dir2.py --dataset ogbl_citation2 --split_ratio $ratio --type txt &
# done

# ogbl-citation2 0.02
# python split_datasets_dir2.py --dataset ogbl_citation2 --split_ratio 0.02 --type pt &
# python split_datasets_dir2.py --dataset twitter --split_ratio 0.02 --type pt