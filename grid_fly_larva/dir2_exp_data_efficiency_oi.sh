cd /home/gaochi/leaf_matching/
dataset=fly_larva
seed=1
cnt=0
devices=(0 1 2 3 4 5 6 7)

for ratio in 0.9; do
    for h in 12 4 6 8 10; do
        for pos_ratio in 0.01 0.05 0.1 0.25 0.6 1.0 1.5 2.0 3.0 5.0 10.0; do
            for gamma in 2 4 6 8; do
                for alpha in 0.1 0.3 0.5 0.8  1 3 6 10  15 20 40 60  80 120 180 240; do
                    device_cnt=$((cnt % 8))  # Use a different GPU for each run
                    device=${devices[$device_cnt]}
                    cnt=$((cnt + 1))

                    CUDA_VISIBLE_DEVICES=$device nohup python dir_main_oi_compact_speedup.py --dataset $dataset --split_ratio $ratio\
                    --h $h --tp 16 --c 2 --neg 1 --gamma $gamma --pos_ratio $pos_ratio --alpha $alpha --epoch 50 --batch_edge 8192 \
                    --seed $seed --chunks 1 --convergence 0.8 --eval_step 10\
                    > ./grid_fly_larva/logs_1/r${ratio}-h${h}-pr${pos_ratio}-g${gamma}-a${alpha}.log 2>&1 &
                    
                    echo "Started processing dataset: $dataset, ratio: $ratio, h: $h, gamma: $gamma, alpha: $alpha on device: $device"
                    

                done
                wait
            done
        done  
    done
done
