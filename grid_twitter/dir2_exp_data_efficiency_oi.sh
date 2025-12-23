cd /home/gaochi/leaf_matching/
dataset=twitter
seed=1
cnt=0
batch_size=40
devices=(0 1 2 3 4 5 6 7)

for ratio in 0.3; do
    for h in 17 15 13; do
        for gamma in 3 4 5; do
            for alpha in 1 5 10 15 20 30 50 80; do
                device_cnt=$((cnt % 8))  # Use a different GPU for each run
                device=${devices[$device_cnt]}
                cnt=$((cnt + 1))

                CUDA_VISIBLE_DEVICES=$device nohup python dir_main_oi.py --dataset $dataset --split_ratio $ratio\
                --h $h --tp 16 --c 3 --neg 1 --gamma $gamma --pos_ratio 1 --alpha $alpha --epoch 50 --batch_size $batch_size \
                --seed $seed --chunks 8 --convergence 0.8 --eval_step 10\
                > ./grid_twitter/logs_1/r${ratio}-h${h}-g${gamma}-a${alpha}.log 2>&1 &


                echo "Started processing dataset: $dataset, ratio: $ratio, h: $h, gamma: $gamma, alpha: $alpha on device: $device"
            done
            wait
        done
    done  
done
