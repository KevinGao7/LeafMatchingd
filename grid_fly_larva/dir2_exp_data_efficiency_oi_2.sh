cd /home/gaochi/leaf_matching/
dataset=fly_larva
seed=1
cnt=0
devices=(0 1 2 3 4 5 6 7)
pos_ratio=2.0
gamma=6
epoch=100

for ratio in 0.9; do
    for h in 16 14 12; do
        for convergence in 0.5 0.65 0.8 0.95; do
            for alpha in 0.1 0.3 0.5 0.8  1 3 6 10  15 20 40 60  80 120 180 240; do
                device_cnt=$((cnt % 8))  # Use a different GPU for each run
                device=${devices[$device_cnt]}
                cnt=$((cnt + 1))

                CUDA_VISIBLE_DEVICES=$device nohup python dir_main_oi_compact_speedup.py --dataset $dataset --split_ratio $ratio\
                --h $h --tp 16 --c 2 --neg 1 --gamma $gamma --pos_ratio $pos_ratio --alpha $alpha --epoch $epoch --batch_edge 8192 \
                --seed $seed --chunks 1 --convergence $convergence --eval_step 10\
                > ./grid_fly_larva/logs_2/h${h}-a${alpha}-con${convergence}.log 2>&1 &
                
                echo "Started processing dataset: $dataset, ratio: $ratio, h: $h, gamma: $gamma, alpha: $alpha, convergence: $convergence on device: $device"
                

            done
            wait
        done
    done
done
