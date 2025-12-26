#!/bin/bash

# making directories
mkdir -p ./logs
export PYTHONPATH=./


# running experiments for synthetic datasets
for power in 10 11 12 13 14 15 16 17 18 19
do
    num_node=$(echo "2^$power" | bc) # 2^power nodes

    dataset='ER_'$num_node'_5'
    logname='scaling_'$num_node

    seed=1
    device=1 # cuda:device

    nohup python undir_main.py --dataset $dataset --device $device \
        --h 16 --tp 16 --neg 1 --c 1 --alpha 16 --gamma 3 --epoch 600 --batch_size 512 --seed $seed \
        > ./logs/$logname-$seed.log 2>&1
    
    wait
done

# running experiments for real-world datasets
for ratio in 0.99 0.5 0.25 0.125 0.0625 0.03125 0.015625
do
    for seed in 1 2 3 4 5
    do
        device=0
        CUDA_VISIBLE_DEVICES=$device nohup python undir_main.py \
            --seed $turn --dataset ogbl_collab --epoch 160 \
            --split_ratio $ratio --alpha 33 --gamma 5 --h 16 --tp 16 --c 3 --neg 1\
            --batch_size 8192 --pos_ratio 0.75\
            > $home_path/collab_logs/$run_name.log 2>&1 &
        cnt=$((cnt+1))
        cnt=$((cnt%${#devices[@]}))

        wait
    done
done