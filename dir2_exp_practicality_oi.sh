#!/bin/bash

# making directories
mkdir -p ./logs_oi
# cnt=0
# devices=(1 2 3 4 5)
# epoch=400

# for dataset in twitter
# do
#     for ratio in 0.02 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#     do
#         for seed in 1
#         do
#             device_cnt=$((cnt % 5))  # Use a different GPU for each run
#             device=${devices[$device_cnt]}
#             cnt=$((cnt + 1))

#             CUDA_VISIBLE_DEVICES=5 nohup python dir_main_oi.py --dataset $dataset --split_ratio $ratio\
#             --h 15 --tp 16 --c 3 --neg 1 --gamma 3 --pos_ratio 1 --alpha 3 --epoch $epoch --batch_size 128 \
#             --seed $seed --chunks 16 --convergence 0.8 --eval_step 10\
#             > ./logs/${dataset}/s${seed}-r${ratio}-e${epoch}.log 2>&1 &

#             echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
#         done
#         wait
#     done
#     echo "$dataset processing completed."
# done


# cnt=0
# devices=(1)
# epoch=100

# for dataset in ogbl_citation2
# do
#     for ratio in 0.02
#     do
#         for seed in 1
#         do
#             device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
#             device=${devices[$device_cnt]}
#             cnt=$((cnt + 1))

#             CUDA_VISIBLE_DEVICES=1 nohup python dir_main_oi.py --dataset $dataset --split_ratio $ratio\
#             --h 15 --tp 16 --c 1 --neg 1 --gamma 3 --pos_ratio 1 --alpha 3 --epoch $epoch --batch_size 512 \
#             --seed $seed --chunks 16 --convergence 0.9 --eval_step 10\
#             > ./logs/${dataset}/new-s${seed}-r${ratio}-e${epoch}.log 2>&1 &

#             echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
#         done
#         wait
#     done
#     echo "$dataset processing completed."
# done




cnt=0
devices=(0 1 2 3 4)

for dataset in ogbl_citation2
do
    for ratio in 0.02
    do
        for seed in 1
        do
            device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
            device=${devices[$device_cnt]}

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
            CUDA_VISIBLE_DEVICES=$device nohup python dir_main_oi_compact_speedup.py --dataset $dataset --split_ratio $ratio\
            --h 15 --tp 16 --c 1 --neg 1 --gamma 3.5 --pos_ratio 1.0 --alpha 8 --epoch 20 --batch_edge 131072 \
            --seed $seed --chunks 16 --convergence 0.5 --eval_step 1 \
            > ./logs_oi/${dataset}/__test__s${seed}-r${ratio}.log 2>&1 &

            cnt=$((cnt + 1))
            if [ $cnt -eq 5 ]; then
                cnt=0
                wait
            fi
        done
    done
    echo "$dataset processing completed."
done

wait