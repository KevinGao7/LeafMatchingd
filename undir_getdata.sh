# #!/bin/bash


# Efficiency
for dataset in cora citeseer pubmed ogbl_collab ogbl_ppa icews18_min icews18_mid icews18_max ro internet kegg; do
        python undir_getdata.py --dataset $dataset --method normal 
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

