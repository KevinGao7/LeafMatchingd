#!/bin/bash
cd /home/gaochi/leaf_matching

# making directories
cnt=0
devices=(2 3 4 5)
seed=1
ratio=0.4
epoch=100
dataset=pubmed
pos_ratio=2.0
convergence=0.8
c=1
h=15


for gamma in 5 6
do
    for alpha_i in 1 2 3
    do
        if [ $alpha_i -eq 1 ]; then
            alphas=(0.1 0.25 0.5 0.8 1 3 8 15)
        elif [ $alpha_i -eq 2 ]; then
            alphas=(27 40 60 80 95 110 125 140) 
        else
            alphas=(160 180 220 260 300 350 400 450)
        fi

        for alpha in ${alphas[@]}
        do
            
            device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
            device=${devices[$device_cnt]}
            cnt=$((cnt + 1))

            CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_oi.py --dataset $dataset --split_ratio $ratio\
            --h ${h} --tp 16 --c ${c} --neg 1 --gamma ${gamma} --pos_ratio ${pos_ratio} --alpha ${alpha} --epoch ${epoch} --batch_size 256 \
            --seed $seed --chunks 16 --convergence ${convergence} --eval_step 10\
            > ./grid_${dataset}/logs_4/s${seed}-h${h}-c${c}-p${pos_ratio}-r${ratio}-g${gamma}-a${alpha}-e${epoch}-con${convergence}.log 2>&1 &

        done
        wait
    done
done
