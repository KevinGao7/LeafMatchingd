#!/bin/bash

# making directories
mkdir -p ./logs
cnt=0
devices=(1)

for dataset in cora_ml
do
    for ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for seed in 1
        do
            device_cnt=$((cnt % 5))  # Use a different GPU for each run
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=1 python dir_main_oi_compact.py --dataset $dataset --split_ratio $ratio\
            --h 13 --tp 16 --c 1 --neg 1 --gamma 3 --pos_ratio 1 --alpha 3 --epoch 50 --batch_size 64 \
            --seed $seed --chunks 8 --convergence 0.8 --eval_step 1

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"

            wait
        done
        wait
    done
    echo "$dataset processing completed."
done
