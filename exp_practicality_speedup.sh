#!/bin/bash

# defining the devices
devices=(2)

# >>> ogbl-collab <<< #
dataset=ogbl_collab
echo "Starting $dataset processing..."
    ratio=0.02
    echo "Processing ratio $ratio ..."
    for turn in 1
    do
        # assign the devices
        device=$turn

        # parallelly start the tasks
        CUDA_VISIBLE_DEVICES=$device python main_speedup.py --dataset $dataset --split_ratio $ratio --turn $turn\
        --h 16 --tp 16 --c 3 --neg 1 --gamma 6 --lam 1.33 --alpha 11 --epoch 5 --batch_size 16384 \
        --seed $turn 
    done

    # waiting for this ratio to be finished
    wait
    echo "Processing ratio $ratio completed."
echo "$dataset processing completed."



# >>> ogbl-ppa <<< #
dataset=ogbl_ppa
echo "Starting $dataset processing..."
    ratio=0.02
    echo "Processing ratio $ratio ..."
    for turn in 1
    do
        # assign the devices
        device=$turn

        # parallelly start the tasks
        CUDA_VISIBLE_DEVICES=$device python main_speedup.py --dataset $dataset --split_ratio $ratio --turn $turn\
        --h 19 --tp 16 --c 1 --neg 1 --gamma 6 --lam 1 --alpha 24 --epoch 5 --batch_size 16384 \
        --seed $turn
    done

    # waiting for this ratio to be finished
    wait
    echo "Processing ratio $ratio completed."
echo "$dataset processing completed."