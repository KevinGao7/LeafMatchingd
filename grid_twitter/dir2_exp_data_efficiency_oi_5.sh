cd /home/gaochi/leaf_matching/
devices=(4 5 6 7)
ratio=0.3

for dataset in twitter
do
    for bs in 4096 8192 16384 32768
    do
        for seed in 1
        do

            device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht_speedup.py --dataset $dataset --split_ratio $ratio\
            --h 14 --tp 16 --c 1 --neg 1 --gamma 5 --pos_ratio 5 --alpha 20 --epoch 120 --batch_size $bs --oi 1\
            --seed $seed --chunks 16 --convergence 0.7 --eval_step 20\
            > ./grid_twitter/logs_5/s${seed}-r${ratio}-bs${bs}.log 2>&1 &

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
        done
    done
    echo "$dataset processing completed."
done
