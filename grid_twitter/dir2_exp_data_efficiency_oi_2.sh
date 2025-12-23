cd /home/gaochi/leaf_matching/
dataset=twitter
seed=1
cnt=0
h=15
gamma=5
alpha=20
devices=(0 1 2 3 4 5 6 7)

for ratio in 0.3; do
    for c in 1 2 3; do
        if [ $c -eq 1 ]; then
            batch_size=128
        elif [ $c -eq 2 ]; then
            batch_size=96
        else
            batch_size=64
        fi

        for convergence in 0.6 0.7 0.8 0.9; do
            for pos_ratio in 0.6 0.8 1.0 1.2; do
                device_cnt=$((cnt % 8))  # Use a different GPU for each run
                device=${devices[$device_cnt]}

                echo "Started processing dataset: $dataset, ratio: $ratio, h: $h, c: $c, gamma: $gamma, pos_ratio: $pos_ratio, alpha: $alpha on device: $device"
                
                CUDA_VISIBLE_DEVICES=$device nohup python dir_main_oi.py --dataset $dataset --split_ratio $ratio\
                --h $h --tp 16 --c $c --neg 1 --gamma $gamma --pos_ratio $pos_ratio --alpha $alpha --epoch 100 --batch_size $bs \
                --seed $seed --chunks 8 --convergence $convergence --eval_step 5\
                > ./grid_twitter/logs_2/r${ratio}_c${c}_conv${convergence}_pr${pos_ratio}_seed${seed}.log 2>&1 &

                cnt=$((cnt + 1))

                if [ $cnt -eq 8 ]; then
                    wait
                    cnt=0
                fi
            done
        done
    done
done
