#!/bin/bash
cd /home/gaochi/leaf_matching

# making directories
cnt=0
devices=(2 3 4 5)
seed=1
ratio=0.4
epoch=300
dataset=pubmed


for convergence in 0.8
do
    for pos_ratio in 1.2 1.5 2.0
    do
        for c in 1
        do
            for h in 12 13 14 15
            do
                for gamma in 6 7 8
                do
                    echo "Processing dataset: $dataset, ratio: $ratio, pos_ratio: $pos_ratio, c: $c, h: $h, gamma: $gamma, convergence: $convergence"
                    for alpha in 0.1 0.25 0.5 0.8 1 3 8 15 27 40 60 80
                    do
                        device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
                        device=${devices[$device_cnt]}
                        cnt=$((cnt + 1))

                        CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_oi.py --dataset $dataset --split_ratio $ratio\
                        --h ${h} --tp 16 --c ${c} --neg 1 --gamma ${gamma} --pos_ratio ${pos_ratio} --alpha ${alpha} --epoch ${epoch} --batch_size 256 \
                        --seed $seed --chunks 16 --convergence ${convergence} --eval_step 10\
                        > ./grid_${dataset}/logs_3/s${seed}-h${h}-c${c}-p${pos_ratio}-r${ratio}-g${gamma}-a${alpha}-e${epoch}-con${convergence}.log 2>&1 &

                    done
                    wait
                done
            done
        done
    done
done
