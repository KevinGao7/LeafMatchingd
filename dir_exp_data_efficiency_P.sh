#!/bin/bash

# making directories
mkdir -p ./logs
cnt=0

# running experiments for cora, citeseer & pubmed
for dataset in pubmed
do
    echo "Starting $dataset processing..."
    for v in -1.0 -0.8 -0.6 -0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0
    do
        echo "Processing v $v for $dataset ..."
        for seed in 1
        do
            device=$((cnt % 8))  # Use a different GPU for each run
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=$device nohup python dir_mainP.py --dataset $dataset --split_ratio 0.9\
            --h 12 --tp 16 --c 1 --neg 1 --v $v --gamma 3 --pos_ratio 1 --alpha 1 --epoch 50 --batch_size 256 \
            --seed $seed --chunks 1 --convergence 0.8 --eval_step 1\
            > ./logs/${dataset}-v${v}-s${seed}.log 2>&1 &

        done
        echo "Processing v $v for $dataset completed."
    done
    echo "$dataset processing completed."
    wait
done

# for dataset in pubmed
# do
#     echo "Starting $dataset processing..."
#     for alpha in 0.125 0.25 0.5 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
#     # for alpha in 8 9 10 11 12 13 14 15 16 17 18
#     do
#         echo "Processing alpha $alpha for $dataset ..."
#         for seed in 1
#         do
#             device=$((cnt % 8))  # Use a different GPU for each run
#             cnt=$((cnt + 1))

#             CUDA_VISIBLE_DEVICES=$device nohup python dir_main_P.py --dataset $dataset --split_ratio 0.9\
#             --h 12 --tp 16 --c 1 --neg 1 --v 0.9 --gamma 3 --pos_ratio 1 --alpha $alpha --epoch 50 --batch_size 256 \
#             --seed $seed --chunks 1 --convergence 0.8 --eval_step 1\
#             > ./logs/${dataset}-alpha222${alpha}-s${seed}.log 2>&1 &

#         done
#         echo "Processing alpha $alpha for $dataset completed."
#     done
#     echo "$dataset processing completed."
#     wait
# done
