# # #!/bin/bash

# # # Cora ML
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_fake.py --dataset cora_ml --split_ratio $ratio --type txt &
#     wait
# done

# # Cora
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_fake.py --dataset cora --split_ratio $ratio --type txt &
#     wait
# done

# # CiteSeer
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_fake.py --dataset citeseer --split_ratio $ratio --type txt &
#     wait
# done

# # PubMed
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_fake.py --dataset pubmed --split_ratio $ratio --type txt &
#     wait
# done
# wait

# # Amazon
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_fake.py --dataset amazon --split_ratio $ratio --type txt &
#     wait
# done
# wait

# # yeast
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_fake.py --dataset yeast --split_ratio $ratio --type txt &
#     wait
# done
# wait

# # male
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_fake.py --dataset male --split_ratio $ratio --type txt &
#     wait
# done
# wait

# ICEWS18
for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python split_datasets_fake.py --dataset icews18_max --split_ratio $ratio --type txt &  
    wait
done

# # twitter
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#     python split_datasets_fake.py --dataset twitter --split_ratio $ratio --type txt &
#     wait
# done
# wait

# # ogbl-citation2
# for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# do
#     python split_datasets_fake.py --dataset ogbl_citation2 --split_ratio $ratio --type txt &
#     wait
# done
