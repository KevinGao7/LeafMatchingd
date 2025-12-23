cd /home/gaochi/leaf_matching/
dataset=icews18_max
seed=1
cnt=0

devices=(7)

for ratio in 0.9; do
    for h in 12; do
        for gamma in 4; do
            for alpha in 19 22 25 28; do
                device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
                device=${devices[$device_cnt]}
                cnt=$((cnt + 1))

                CUDA_VISIBLE_DEVICES=$device nohup python dir_main_ht_speedup.py --dataset $dataset --split_ratio $ratio\
                --h $h --tp 16 --c 1 --neg 1 --gamma $gamma --pos_ratio 1 --alpha $alpha --epoch 50 --batch_size 16384 \
                --seed $seed --chunks 1 --convergence 0.8 --eval_step 5\
                > ./grid_icews18/logs_2/r${ratio}-h${h}-g${gamma}-a${alpha}.log 2>&1 &

                echo "Started processing dataset: $dataset, ratio: $ratio, h: $h, gamma: $gamma, alpha: $alpha on device: $device"
            done
            wait
        done
    done  
done
