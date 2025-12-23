#!/bin/bash

## 注意，cuda devices 被改过了


# making directories
mkdir -p ./logs_ht
cnt=0
devices=(0 1 2 3 4 5 6 7)

# cora_ml
for dataset in cora_ml
do
    for ratio in 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9
    do
        for seed in 1
        do
            device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht.py --dataset $dataset --split_ratio $ratio\
            --h 13 --tp 16 --c 3 --neg 1 --gamma 4 --pos_ratio 1 --alpha 2 --epoch 75 --batch_size 256 --oi 0\
            --seed $seed --chunks 1 --convergence 0.8 --eval_step 1\
            > ./logs_ht/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        # wait
    done
    echo "$dataset processing completed."
done

wait

# # citeseer
for dataset in citeseer
do
    for ratio in 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9
    do
        for seed in 1
        do
            device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht.py --dataset $dataset --split_ratio $ratio\
            --h 13 --tp 16 --c 1 --neg 1 --gamma 4 --pos_ratio 1 --alpha 15 --epoch 100 --batch_size 256 --oi 0\
            --seed $seed --chunks 1 --convergence 0.8 --eval_step 1\
            > ./logs_ht/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        # wait
    done
    echo "$dataset processing completed."
done

wait

# cora
for dataset in cora
do
    for ratio in 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9
    do
        for seed in 1
        do
            device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht.py --dataset $dataset --split_ratio $ratio\
            --h 14 --tp 16 --c 1 --neg 1 --gamma 8 --pos_ratio 1 --alpha 180 --epoch 100 --batch_size 256 --oi 0\
            --seed $seed --chunks 16 --convergence 0.8 --eval_step 10\
            > ./logs_ht/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        # wait
    done
    echo "$dataset processing completed."
done

wait

# pubmed
for dataset in pubmed
do
    for ratio in 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9
    do
        for seed in 1
        do
            device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht.py --dataset $dataset --split_ratio $ratio\
            --h 12 --tp 16 --c 1 --neg 1 --gamma 5 --pos_ratio 1.5 --alpha 3 --epoch 75 --batch_size 128 --oi 0\
            --seed $seed --chunks 8 --convergence 0.8 --eval_step 5\
            > ./logs_ht/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        # wait
    done
    echo "$dataset processing completed."
done

wait

## 仅保留了 icews18，且种子只有 1 个，同时运行没有 wait
## h=13.0, gamma=6.0, alpha=15.0
# devices=(0 1 2 3 4 5 6 7)
# cnt=0
for dataset in icews18_max
do
    for seed in 1
    do
        for ratio in 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9
        do
                device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
                device=${devices[$device_cnt]}
                cnt=$((cnt + 1))

                CUDA_VISIBLE_DEVICES=$device nohup python dir_main_ht.py --dataset $dataset --split_ratio $ratio\
                --h 13 --tp 16 --c 1 --neg 1 --gamma 6 --pos_ratio 1 --alpha 15 --epoch 50 --batch_size 64 --oi 0\
                --seed $seed --chunks 8 --convergence 0.8 --eval_step 1\
                > ./logs_ht/${dataset}/s${seed}-r${ratio}.log 2>&1 &

                echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"

                # wait
        done
        # wait
    done
    echo "$dataset processing completed."
done

wait

# twitter
_batch_size=64
for dataset in twitter
do
    for ratio in 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9
    do
        for seed in 1
        do
            if (( $(echo "$ratio > 0.6" | bc -l) )); then
                _batch_size=32
            else
                _batch_size=64
            fi
            device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht.py --dataset $dataset --split_ratio $ratio\
            --h 15 --tp 16 --c 1 --neg 1 --gamma 5 --pos_ratio 5 --alpha 20 --epoch 100 --batch_size ${_batch_size} --oi 0\
            --seed $seed --chunks 16 --convergence 0.7 --eval_step 5\
            > ./logs_ht/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        # wait
    done
    echo "$dataset processing completed."
done

wait