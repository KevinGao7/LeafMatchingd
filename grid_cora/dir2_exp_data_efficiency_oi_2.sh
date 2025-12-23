#!/bin/bash
cd /home/gaochi/leaf_matching

# making directories
cnt=0
devices=(2 3 4 5)
seed=1
c=1
ratio=0.4
epoch=100
dataset=cora
batch_size=128
convergence=0.8
pos_ratio=1
batch_size=256


for h in 14 15
do
    for gamma in 6 7 8
    do
        echo "Processing dataset: $dataset, ratio: $ratio, h: $h, gamma: $gamma"
        for alpha in 30 40 50 60 75 90 105 120 140 160 180 200
        do
            device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_oi.py --dataset $dataset --split_ratio $ratio\
            --h ${h} --tp 16 --c ${c} --neg 1 --gamma ${gamma} --pos_ratio ${pos_ratio} --alpha ${alpha} --epoch ${epoch} --batch_size ${batch_size} \
            --seed $seed --chunks 16 --convergence ${convergence} --eval_step 10\
            > ./grid_${dataset}/logs_2/s${seed}-h${h}-c${c}-p${pos_ratio}-r${ratio}-g${gamma}-a${alpha}-e${epoch}-con${convergence}.log 2>&1 &
        done
        wait
    done
done