#!/bin/bash

# making directories
mkdir -p ./logs
cnt=0

# running experiments for cora, citeseer & pubmed
# for dataset in cora_ml citeseer pubmed
# do
#     echo "Starting $dataset processing..."
#     for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#     do
#         echo "Processing ratio $ratio for $dataset ..."
#         for seed in 1
#         do
#             device=$((cnt % 8))  # Use a different GPU for each run
#             cnt=$((cnt + 1))

#             CUDA_VISIBLE_DEVICES=$device nohup python dir_main.py --dataset $dataset --split_ratio $ratio\
#             --h 12 --tp 16 --c 1 --neg 1 --gamma 3 --pos_ratio 1 --alpha 8 --epoch 50 --batch_size 256 \
#             --seed $seed --chunks 1 --convergence 0.8 --eval_step 1\
#             > ./logs/${dataset}-r${ratio}-s${seed}.log 2>&1 &

#         done
#         echo "Processing ratio $ratio for $dataset completed."
#     done
#     echo "$dataset processing completed."
#     wait

for dataset in pubmed
do
    echo "Starting $dataset processing..."
    for alpha in 0.125 0.25 0.5 1 2 3 4 5 6 7 8
    do
        echo "Processing alpha $alpha for $dataset ..."
        for seed in 1
        do
            device=$((cnt % 8))  # Use a different GPU for each run
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=$device nohup python dir_main.py --dataset $dataset --split_ratio 0.9\
            --h 12 --tp 16 --c 1 --neg 1 --gamma 3 --pos_ratio 1 --alpha $alpha --epoch 50 --batch_size 256 \
            --seed $seed --chunks 1 --convergence 0.8 --eval_step 1\
            > ./logs/${dataset}-alpha000${alpha}-s${seed}.log 2>&1 &

        done
        echo "Processing alpha $alpha for $dataset completed."
    done
    echo "$dataset processing completed."
    wait
done
