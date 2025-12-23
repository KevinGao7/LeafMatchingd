cd /home/gaochi/leaf_matching/
dataset=twitter
seed=1
cnt=0
h=15
gamma=5
c=1
bs=128
pos_ratio=5.0
devices=(0 1 2 3 4 5 6 7)

for ratio in 0.3; do
    for convergence in 0.67 0.70 0.73 0.75; do
        for alpha in 15 16 17 17.5 18 18.5 19 19.5 20 20.5 21 21.5 22 22.5 23 24; do
            device_cnt=$((cnt % 8))  # Use a different GPU for each run
            device=${devices[$device_cnt]}

            echo "Started processing dataset: $dataset, ratio: $ratio, h: $h, c: $c, gamma: $gamma, pos_ratio: $pos_ratio, alpha: $alpha on device: $device"

            CUDA_VISIBLE_DEVICES=$device nohup python dir_main_oi.py --dataset $dataset --split_ratio $ratio\
            --h $h --tp 16 --c $c --neg 1 --gamma $gamma --pos_ratio $pos_ratio --alpha $alpha --epoch 100 \
            --batch_size $bs \
            --seed $seed --chunks 8 --convergence $convergence --eval_step 5\
            > ./grid_twitter/logs_3/r${ratio}_conv${convergence}_alpha${alpha}.log 2>&1 &

            cnt=$((cnt + 1))

            if [ $cnt -eq 8 ]; then
                wait
                cnt=0
            fi
        done
    done
done
