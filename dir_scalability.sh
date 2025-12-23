#!/bin/bash

# making directories
mkdir -p ./logs_scalability
mkdir -p ./logs_scalability/synthetic
mkdir -p ./logs_scalability/realworlds
mkdir -p ./logs_scalability/synthetic_ht
mkdir -p ./logs_scalability/realworlds_ht
export PYTHONPATH=./
devices=(0 1 2 3 4)
cnt=0

# >>>>>>>>>> sr <<<<<<<<<< #

# running experiments for synthetic datasets
for power in 10 11 12 13 14 15 16 17 18 19
do
    for seed in 1 2 3 4 5
    do
        num_node=$(echo "2^$power" | bc) # 2^power nodes

        dataset='ER_'$num_node'_5'
        logname='scaling_'$num_node

        device=${devices[$((cnt%${#devices[@]}))]}
        cnt=$((cnt+1))

        echo "Processing dataset $dataset seed $seed at device $device"
        CUDA_VISIBLE_DEVICES=${device} python dir_main_sr.py --dataset $dataset \
            --h 16 --tp 16 --neg 1 --c 1 --alpha 16 --gamma 3 --epoch 300 --batch_edge 8192 --seed $seed --convergence 0.0\
            > ./logs_scalability/synthetic/${logname}_s${seed}.log 2>&1 &

        if [ $((cnt % ${#devices[@]})) -eq 0 ]; then
            wait
        fi
    done
done

# running experiments for real-world datasets
for seed in 1 2 3 4 5
do
    for ratio in 0.99 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625
    do
        device=${devices[$((cnt%${#devices[@]}))]}
        cnt=$((cnt+1))

        echo "Processing ratio $ratio seed $seed at device $device"
        CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_oi_compact_speedup.py --dataset twitter --split_ratio $ratio\
                    --h 16 --tp 16 --c 1 --neg 1 --gamma 5 --pos_ratio 5 --alpha 20 --epoch 250 --batch_edge 16384 \
                    --seed $seed --chunks 16 --convergence 0.0 --eval_step 500\
                    > ./logs_scalability/realworlds/scaling_r${ratio}_s${seed}.log 2>&1 &
        if [ $((cnt % ${#devices[@]})) -eq 0 ]; then
            wait
        fi
    done
    wait
done












### >>> ht <<< ###



running experiments for synthetic datasets
for power in 10 11 12 13 14 15 16 17 18 19
do
    for seed in 1 2 3 4 5
    do
        num_node=$(echo "2^$power" | bc) # 2^power nodes

        dataset='ER_'$num_node'_5'
        logname='scaling_'$num_node

        device=${devices[$((cnt%${#devices[@]}))]}
        cnt=$((cnt+1))

        echo "Processing dataset $dataset seed $seed at device $device"
        CUDA_VISIBLE_DEVICES=${device} python dir_main_ht_speedup.py --dataset $dataset \
            --h 16 --tp 16 --neg 1 --c 1 --alpha 16 --gamma 3 --epoch 300 --batch_size 8192 --seed $seed --convergence 1.0\
            > ./logs_scalability/synthetic_ht/${logname}_s${seed}.log 2>&1 &

        if [ $((cnt % ${#devices[@]})) -eq 0 ]; then
            wait
        fi
    done
done

running experiments for real-world datasets


# for ratio in 0.99 # 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625
#     do
#     for seed in 1 2 3 4 5
#     do
#         device=${devices[$((cnt%${#devices[@]}))]}
#         cnt=$((cnt+1))

#         echo "Processing ratio $ratio seed $seed at device $device"
#         CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht_speedup.py --dataset twitter --split_ratio $ratio\
#                     --h 16 --tp 16 --c 1 --neg 1 --gamma 5 --pos_ratio 5 --alpha 20 --epoch 250 --batch_size 16384 \
#                     --seed $seed --chunks 16 --convergence 0.0 --eval_step 500\
#                     > ./logs_scalability/realworlds_ht/scaling_r${ratio}_s${seed}.log 2>&1 &
#         if [ $((cnt % ${#devices[@]})) -eq 0 ]; then
#             wait
#         fi
#     done
#     wait
# done