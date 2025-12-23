# making directories
cnt=0
devices=(5 6 7)

# # cora_ml
# for dataset in cora_ml
# do
#     for ratio in 0.3 0.6 0.9
#     do
#         for seed in 1
#         do
#             mkdir -p ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}

#             device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
#             device=${devices[$device_cnt]}
#             cnt=$((cnt + 1))

#             CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht_speedup.py --dataset $dataset --split_ratio $ratio\
#             --h 13 --tp 16 --c 1 --neg 1 --gamma 4 --pos_ratio 1 --alpha 15 --epoch 150 --batch_size 8192 \
#             --seed $seed --chunks 1 --convergence 0.8 --eval_step 1 --oi 1 --save_score ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio} --save_emb ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/emb.pt\
#             > ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/output.log 2>&1 &

#             echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
#         done
#     done
#     echo "$dataset processing completed."
# done

# wait

# # citeseer
# for dataset in citeseer
# do
#     for ratio in 0.3 0.6 0.9
#     do
#         for seed in 1
#         do

#             mkdir -p ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}
#             device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
#             device=${devices[$device_cnt]}
#             cnt=$((cnt + 1))

#             CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht_speedup.py --dataset $dataset --split_ratio $ratio\
#             --h 13 --tp 16 --c 1 --neg 1 --gamma 4 --pos_ratio 1 --alpha 15 --epoch 150 --batch_size 8192 \
#             --seed $seed --chunks 1 --convergence 0.8 --eval_step 1 --oi 1 --save_score ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio} --save_emb ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/emb.pt\
#             > ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/output.log 2>&1 &

#             echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
#         done
#     done
#     echo "$dataset processing completed."
# done

# wait

# # cora
# for dataset in cora
# do
#     for ratio in 0.3 0.6 0.9
#     do
#         for seed in 1
#         do
#             mkdir -p ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}
#             device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
#             device=${devices[$device_cnt]}
#             cnt=$((cnt + 1))

#             CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht_speedup.py --dataset $dataset --split_ratio $ratio\
#             --h 14 --tp 16 --c 1 --neg 1 --gamma 8 --pos_ratio 1 --alpha 180 --epoch 100 --batch_size 8192 \
#             --seed $seed --chunks 16 --convergence 0.8 --eval_step 10 --oi 1 --save_score ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio} --save_emb ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/emb.pt\
#             > ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/output.log 2>&1 &

#             echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
#         done
#     done
#     echo "$dataset processing completed."
# done

# wait

# icews18_max
for dataset in icews18_max
do        
    for ratio in 0.3 0.6 0.9
    do
        for seed in 1
        do
            mkdir -p ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}
            device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=$device nohup python dir_main_ht_speedup.py --dataset $dataset --split_ratio $ratio\
            --h 12 --tp 16 --c 1 --neg 1 --gamma 4 --pos_ratio 1 --alpha 18 --epoch 100 --batch_size 16384 --oi 1\
            --seed $seed --chunks 8 --convergence 0.8 --eval_step 10 --save_score ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio} --save_emb ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/emb.pt\
            > ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/output.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"

        done
        wait
    done
    echo "$dataset processing completed."
done

wait

# for dataset in twitter
# do
#     for ratio in 0.3 0.6 0.9
#     do
#         for seed in 1
#         do
#             mkdir -p ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}
#             device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
#             device=${devices[$device_cnt]}
#             cnt=$((cnt + 1))

#             CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht_speedup.py --dataset $dataset --split_ratio $ratio\
#             --h 15 --tp 16 --c 1 --neg 1 --gamma 5 --pos_ratio 5 --alpha 20 --epoch 100 --batch_size 8192 \
#             --seed $seed --chunks 16 --convergence 0.7 --eval_step 10 --oi 1 --save_score ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio} --save_emb ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/emb.pt\
#             > ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/output.log 2>&1 &

#             echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
#         done
#     done
#     echo "$dataset processing completed."
# done

# wait

# # male
# cnt=0
# for dataset in male
# do
#     for ratio in 0.3 0.6 0.9
#     do
#         for seed in 1
#         do
#             mkdir -p ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}
#             device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
#             device=${devices[$device_cnt]}
#             cnt=$((cnt + 1))

#             CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht_speedup.py --dataset $dataset --split_ratio $ratio\
#             --h 12 --tp 16 --c 1 --neg 1 --gamma 5 --pos_ratio 8 --alpha 40 --epoch 100 --batch_size 16384 \
#             --seed $seed --chunks 1 --convergence 0.8 --eval_step 10 --oi 1 --save_score ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio} --save_emb ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/emb.pt\
#             > ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/output.log 2>&1 &

#             echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
#         done
#     done
#     echo "$dataset processing completed."
# done

# wait

# # yeast
# cnt=0
# for dataset in yeast
# do
#     for ratio in 0.3 0.6 0.9
#     do
#         for seed in 1
#         do
#             mkdir -p ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}
#             device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
#             device=${devices[$device_cnt]}
#             cnt=$((cnt + 1))

#             CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht_speedup.py --dataset $dataset --split_ratio $ratio\
#             --h 13 --tp 16 --c 1 --neg 1 --gamma 4 --pos_ratio 1 --alpha 15 --epoch 100 --batch_size 16384 \
#             --seed $seed --chunks 1 --convergence 0.8 --eval_step 10 --oi 1 --save_score ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio} --save_emb ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/emb.pt\
#             > ./logs_ht_oi/${dataset}/scores/s${seed}-r${ratio}/output.log 2>&1 &

#             echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
#         done
#     done
#     echo "$dataset processing completed."
# done
