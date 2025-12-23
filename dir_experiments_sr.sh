#!/bin/bash

mkdir -p ./logs_sr
devices=(0 1 2 3 4)


cnt=0
for dataset in male
do
    mkdir -p ./logs_sr/${dataset}
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for seed in 1 2 3 4 5
        do
            device_cnt=$((cnt % ${#devices[@]}))  
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_sr.py --dataset $dataset --split_ratio $ratio\
            --h 12 --tp 16 --c 1 --neg 1 --gamma 5 --pos_ratio 8 --alpha 40 --epoch 100 --batch_edge 16384 \
            --seed $seed --chunks 1 --convergence 0.8 --eval_step 1\
            > ./logs_sr/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        wait
    done
    echo "$dataset processing completed."
done


cnt=0
for dataset in yeast
do
    mkdir -p ./logs_sr/${dataset}
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for seed in 1 2 3 4 5
        do
            device_cnt=$((cnt % ${#devices[@]}))  
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_sr.py --dataset $dataset --split_ratio $ratio\
            --h 13 --tp 16 --c 1 --neg 1 --gamma 4 --pos_ratio 1 --alpha 15 --epoch 100 --batch_edge 16384 \
            --seed $seed --chunks 1 --convergence 0.8 --eval_step 1\
            > ./logs_sr/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        wait
    done
    echo "$dataset processing completed."
done


cnt=0
for dataset in cora_ml
do
    mkdir -p ./logs_sr/${dataset}
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for seed in 1 2 3 4 5
        do
            device_cnt=$((cnt % ${#devices[@]}))  
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_sr.py --dataset $dataset --split_ratio $ratio\
            --h 13 --tp 16 --c 3 --neg 1 --gamma 4 --pos_ratio 1 --alpha 2 --epoch 100 --batch_edge 16384 \
            --seed $seed --chunks 1 --convergence 0.8 --eval_step 1\
            > ./logs_sr/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        wait
    done
    echo "$dataset processing completed."
done


cnt=0
for dataset in citeseer
do
    mkdir -p ./logs_sr/${dataset}
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for seed in 1 2 3 4 5
        do
            device_cnt=$((cnt % ${#devices[@]}))  
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_sr.py --dataset $dataset --split_ratio $ratio\
            --h 13 --tp 16 --c 1 --neg 1 --gamma 4 --pos_ratio 1 --alpha 15 --epoch 100 --batch_edge 16384 \
            --seed $seed --chunks 1 --convergence 0.8 --eval_step 1\
            > ./logs_sr/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        wait
    done
    echo "$dataset processing completed."
done




cnt=0
for dataset in cora
do
    mkdir -p ./logs_sr/${dataset}
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for seed in 1 2 3 4 5
        do
            device_cnt=$((cnt % ${#devices[@]}))  
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_sr.py --dataset $dataset --split_ratio $ratio\
            --h 14 --tp 16 --c 1 --neg 1 --gamma 8 --pos_ratio 1 --alpha 180 --epoch 100 --batch_edge 16384 \
            --seed $seed --chunks 16 --convergence 0.8 --eval_step 10\
            > ./logs_sr/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        wait
    done
    echo "$dataset processing completed."
done




cnt=0
for dataset in pubmed
do
    mkdir -p ./logs_sr/${dataset}
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for seed in 1 2 3 4 5
        do
            device_cnt=$((cnt % ${#devices[@]}))  
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_sr.py --dataset $dataset --split_ratio $ratio\
            --h 12 --tp 16 --c 1 --neg 1 --gamma 5 --pos_ratio 1.5 --alpha 3 --epoch 100 --batch_edge 16384 \
            --seed $seed --chunks 8 --convergence 0.8 --eval_step 5\
            > ./logs_sr/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        wait
    done
    echo "$dataset processing completed."
done



cnt=0
for dataset in amazon
do
    mkdir -p ./logs_sr/${dataset}
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for seed in 1 2 3 4
        do

            device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_sr.py --dataset $dataset --split_ratio $ratio\
            --h 17 --tp 16 --c 1 --neg 1 --gamma 5 --pos_ratio 5 --alpha 20 --epoch 100 --batch_edge 16384 \
            --seed $seed --chunks 16 --convergence 0.7 --eval_step 5\
            > ./logs_sr/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        wait
    done
    echo "$dataset processing completed."
done



cnt=0
for dataset in icews18_min icews18_mid icews18_max
do  
    mkdir -p ./logs_sr/${dataset}
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for seed in 1 2 3 4 5
        do
            device_cnt=$((cnt % ${#devices[@]})) 
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=$device nohup python dir_main_sr.py --dataset $dataset --split_ratio $ratio\
            --h 12 --tp 16 --c 1 --neg 1 --gamma 4 --pos_ratio 1 --alpha 18 --epoch 100 --batch_edge 16384 \
            --seed $seed --chunks 8 --convergence 0.8 --eval_step 10\
            > ./logs_sr/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"

        done
        wait
    done
    echo "$dataset processing completed."
done



cnt=0
for dataset in twitter
do
    mkdir -p ./logs_sr/${dataset}
    for ratio in 0.02 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for seed in 1 2 3 4 5
        do

            device_cnt=$((cnt % ${#devices[@]}))
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_sr.py --dataset $dataset --split_ratio $ratio\
            --h 15 --tp 16 --c 1 --neg 1 --gamma 5 --pos_ratio 5 --alpha 20 --epoch 100 --batch_edge 16384 \
            --seed $seed --chunks 16 --convergence 0.7 --eval_step 5\
            > ./logs_sr/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
        wait
    done
    echo "$dataset processing completed."
done



cnt=0
for dataset in ogbl_citation2
do
    mkdir -p ./logs_sr/${dataset}
    for ratio in 0.02 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for seed in 1 2 3 4 5
        do
            device_cnt=$((cnt % ${#devices[@]}))
            device=${devices[$device_cnt]}

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
            CUDA_VISIBLE_DEVICES=$device nohup python dir_main_sr.py --dataset $dataset --split_ratio $ratio\
            --h 15 --tp 16 --c 1 --neg 1 --gamma 6 --pos_ratio 1 --alpha 45 --epoch 60 --batch_edge 16384 \
            --seed $seed --chunks 16 --convergence 0.6 --eval_step 10 \
            > ./logs_sr/${dataset}/s${seed}-r${ratio}.log 2>&1 &

            cnt=$((cnt + 1))
        done
        wait
    done
    echo "$dataset processing completed."
done