# #!/bin/bash


# Efficiency
for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    for dataset in cora_ml cora citeseer pubmed amazon yeast male icews18_min icews18_mid icews18_max twitter ogbl_citation2; do
        python dir_getdata.py --dataset $dataset --split_ratio $ratio --type txt 
done


# Scalability
for ratio in 0.99 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625; do
    python dir_getdata_sca-real.py --dataset twitter --split_ratio $ratio --type pt &
done
python dir_getdata_sca-syn.py &


# Practicality
for ratio in 0.02; do
    python dir_getdata.py --dataset ogbl_citation2 --split_ratio 0.02 --type pt &
    python dir_getdata.py --dataset twitter --split_ratio 0.02 --type pt
done

