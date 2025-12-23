#!/bin/bash

## 注意，cuda devices 被改过了


# making directories
mkdir -p ./speedup_logs
cnt=0
devices=(0)

# cora_ml
for dataset in cora
do
    for ratio in 0.9
    do
        for seed in 1
        do
            device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_oi_compact_speedup.py --dataset $dataset --split_ratio $ratio\
            --h 13 --tp 16 --c 1 --neg 1 --gamma 4 --pos_ratio 1 --alpha 15 --epoch 150 --batch_edge 32 \
            --seed $seed --chunks 1 --convergence 0.8 --eval_step 1 > ./speedup_logs/${dataset}/log_ratio_${ratio}_seed_${seed}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        wait
    done
    echo "$dataset processing completed."
done