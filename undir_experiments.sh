#!/bin/bash

# making directories
mkdir -p ./logs
mkdir -p ./results

devices=(0 1 2 3 4)


cnt=0
for dataset in icews18_min icews18_mid icews18_max
do
    echo "Starting $dataset processing..."
    mkdir -p ./logs/${dataset}
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        echo "Processing ratio $ratio for $dataset ..."
        for turn in 1 2 3 4 5
        do
            # assign the devices
            device=${devices[$((cnt % ${#devices[@]}))]}
            cnt=$((cnt + 1))

            # parallelly start the tasks
            CUDA_VISIBLE_DEVICES=$device nohup python undir_main.py --dataset $dataset --split_ratio $ratio --turn $turn\
            --h 14 --tp 16 --c 1 --neg 1 --gamma 3 --lam 1.0 --alpha 33 --epoch 50 --batch_size 8192 \
            --seed $turn \
            > ./logs/${dataset}/r${ratio}_s${turn}.log 2>&1 &

            wait

            echo "Started turn $turn for ratio $ratio on device $device."
        done
        # waiting for this ratio to be finished
        echo "Processing ratio $ratio for $dataset completed."
        wait
    done
    echo "$dataset processing completed."
    wait
done



cnt=0
for dataset in kegg
do
    echo "Starting $dataset processing..."
    mkdir -p ./logs/${dataset}
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        echo "Processing ratio $ratio for $dataset ..."
        for turn in 1 2 3 4 5
        do
            # assign the devices
            device=${devices[$((cnt % ${#devices[@]}))]}
            cnt=$((cnt + 1))

            # parallelly start the tasks
            CUDA_VISIBLE_DEVICES=$device nohup python undir_main.py --dataset $dataset --split_ratio $ratio --turn $turn\
            --h 14 --tp 16 --c 1 --neg 1 --gamma 3 --lam 1.0 --alpha 8 --epoch 150 --batch_size 8192 \
            --seed $turn \
            > ./logs/${dataset}/r${ratio}_s${turn}.log 2>&1 &

            echo "Started turn $turn for ratio $ratio on device $device."
        done
        # waiting for this ratio to be finished
        echo "Processing ratio $ratio for $dataset completed."
        wait
    done
    echo "$dataset processing completed."
    wait
done



cnt=0
for dataset in ro
do
    echo "Starting $dataset processing..."
    mkdir -p ./logs/${dataset}
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        echo "Processing ratio $ratio for $dataset ..."
        for turn in 1 2 3 4 5
        do
            # assign the devices
            device=${devices[$((cnt % ${#devices[@]}))]}
            cnt=$((cnt + 1))

            # parallelly start the tasks
            CUDA_VISIBLE_DEVICES=$device nohup python undir_main.py --dataset $dataset --split_ratio $ratio --turn $turn\
            --h 14 --tp 16 --c 1 --neg 1 --gamma 3 --lam 1.5 --alpha 8 --epoch 50 --batch_size 8192 \
            --seed $turn \
            > ./logs/${dataset}/r${ratio}_s${turn}.log 2>&1 &

            echo "Started turn $turn for ratio $ratio on device $device."
        done
        # waiting for this ratio to be finished
        echo "Processing ratio $ratio for $dataset completed."
        wait
    done
    echo "$dataset processing completed."
    wait
done


cnt=0
for dataset in cora citeseer pubmed internet kegg
do
    echo "Starting $dataset processing..."
    mkdir -p ./logs/${dataset}
    for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        echo "Processing ratio $ratio for $dataset ..."
        for turn in 1 2 3 4 5
        do
            # assign the devices
            device=${devices[$((cnt % ${#devices[@]}))]}
            cnt=$((cnt + 1))

            # parallelly start the tasks
            CUDA_VISIBLE_DEVICES=$device nohup python undir_main.py --dataset $dataset --split_ratio $ratio --turn $turn\
            --h 12 --tp 16 --c 1 --neg 1 --gamma 3 --lam 0.3 --alpha 8 --epoch 50 --batch_size 8192 \
            --seed $turn \
            > ./logs/${dataset}/r${ratio}_s${turn}.log 2>&1 &

            echo "Started turn $turn for ratio $ratio on device $device."
        done
        # waiting for this ratio to be finished
        echo "Processing ratio $ratio for $dataset completed."
        wait
    done
    echo "$dataset processing completed."
    wait
done

# evaluation
python eval-speedup.py