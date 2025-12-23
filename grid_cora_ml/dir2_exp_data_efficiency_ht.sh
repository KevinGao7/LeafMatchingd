#!/bin/bash
cd /home/gaochi/leaf_matching

# making directories
cnt=0
devices=(0 1)
seed=1
ratio=0.9
epoch=75
dataset=cora_ml
c=1

for pos_ratio in 0.2 1 5.0 10
do
    for h in 12 13 14 15
    do
        for gamma in 2 3 4 5 6 7
        do
            echo "Running with pos_ratio=${pos_ratio}, c=${c}, h=${h}, gamma=${gamma}"
            for alpha in 0.1 0.5 1 5 10 15 24 36 50 74 100 120 150 200 250 300
            do
                device_cnt=$((cnt % ${#devices[@]}))  # Use a different GPU for each run
                device=${devices[$device_cnt]}
                

                CUDA_VISIBLE_DEVICES=${device} nohup python dir_main_ht.py --dataset $dataset --split_ratio $ratio\
                --h ${h} --tp 16 --c ${c} --neg 1 --gamma ${gamma} --pos_ratio ${pos_ratio} --alpha ${alpha} --epoch ${epoch} --batch_size 128 \
                --seed $seed --chunks 16 --convergence 0.8 --eval_step 5\
                > ./grid_cora_ml/logs_2/s${seed}-h${h}-c${c}-pr${pos_ratio}-r${ratio}-g${gamma}-a${alpha}.log 2>&1 &

                cnt=$((cnt + 1))
                if [ ${cnt} -eq 8 ]; then
                    wait
                    cnt=0
                fi
                
            done
            wait
        done
    done
done
