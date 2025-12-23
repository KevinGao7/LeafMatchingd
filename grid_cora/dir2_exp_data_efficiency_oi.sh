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

sleep 18000

for pos_ratio in 1.5 1 0.5
do
    for h in 14 15 16
    do
        if [ $h -eq 14 ] || [ $h -eq 15 ]; then
            batch_size=256
        else
            batch_size=128
        fi

        for gamma in 3 4 5 6
        do
            echo "Processing dataset: $dataset, ratio: $ratio, pos_ratio: $pos_ratio, c: $c, h: $h, gamma: $gamma, convergence: $convergence"
            for alpha in 0.1 0.5 1 3 8 15 27 40
            do
                device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
                device=${devices[$device_cnt]}
                cnt=$((cnt + 1))

                CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_oi.py --dataset $dataset --split_ratio $ratio\
                --h ${h} --tp 16 --c ${c} --neg 1 --gamma ${gamma} --pos_ratio ${pos_ratio} --alpha ${alpha} --epoch ${epoch} --batch_size ${batch_size} \
                --seed $seed --chunks 16 --convergence ${convergence} --eval_step 5\
                > ./grid_${dataset}/logs_1/s${seed}-h${h}-c${c}-p${pos_ratio}-r${ratio}-g${gamma}-a${alpha}-e${epoch}-con${convergence}.log 2>&1 &

            done
            wait
        done
    done
done