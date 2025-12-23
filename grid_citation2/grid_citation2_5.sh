# 156
cnt=0
devices=(0 1 2 3 4 5)
epoch=60
ratio=0.02
dataset=ogbl_citation2

# gamma=3.5, epoch=30, alpha=8, pos_ratio=1.2, convergence=1.0, c=1

for gamma in 6 7 8; do # 5
    for pos_ratio in 1.0; do
        for alpha in 1 3 6  10 15 21  30 45 60   80 110 150; do 
            device_cnt=$((cnt % ${#devices[@]}))  
            device=${devices[$device_cnt]}

            echo "Started processing dataset: $dataset, ratio: $ratio, seed: $seed on device: $device"
            CUDA_VISIBLE_DEVICES=$device nohup python dir_main_oi_compact_speedup.py --dataset $dataset --split_ratio $ratio\
            --h 15 --tp 16 --c 1 --neg 1 --gamma $gamma --pos_ratio $pos_ratio --alpha $alpha --epoch $epoch --batch_edge 131072 \
            --seed 1 --chunks 16 --convergence 0.6 --eval_step 10 \
            > ./grid_citation2/logs_5/g${gamma}-pr${pos_ratio}-a${alpha}.log 2>&1 &

            cnt=$((cnt + 1))
            if [ $cnt -eq ${#devices[@]} ]; then
                cnt=0
                wait
            fi
        done
    done
done
echo "$dataset processing completed."
