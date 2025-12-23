cd /home/gaochi/leaf_matching/
dataset=twitter
seed=1
cnt=0
h=15
gamma=5
c=1
bs=128
pos_ratio=5.0
alpha=16
devices=(0 1 2 3 4 5 6 7)

for ratio in 0.3; do
    for convergence in 0.70 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.80 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.90 0.91 0.92 0.93; do
        device_cnt=$((cnt % 8))  # Use a different GPU for each run
        device=${devices[$device_cnt]}

        echo "Started processing dataset: $dataset, ratio: $ratio, h: $h, c: $c, gamma: $gamma, pos_ratio: $pos_ratio, alpha: $alpha on device: $device"

        CUDA_VISIBLE_DEVICES=$device nohup python dir_main_oi.py --dataset $dataset --split_ratio $ratio\
        --h $h --tp 16 --c $c --neg 1 --gamma $gamma --pos_ratio $pos_ratio --alpha $alpha --epoch 100 \
        --batch_size $bs \
        --seed $seed --chunks 8 --convergence $convergence --eval_step 5\
        > ./grid_twitter/logs_4/r${ratio}_conv${convergence}.log 2>&1 &

        cnt=$((cnt + 1))

        if [ $cnt -eq 8 ]; then
            wait
            cnt=0
        fi
    done
done
