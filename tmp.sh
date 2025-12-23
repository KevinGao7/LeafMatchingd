# CUDA_VISIBLE_DEVICES=7 python dir_main_ht_speedup.py --dataset twitter --split_ratio 0.02\
#             --h 15 --tp 16 --c 1 --neg 1 --gamma 5 --pos_ratio 5 --alpha 20 --epoch 100 --batch_size 16384 \
#             --seed 0 --chunks 16 --convergence 0.7 --eval_step 5

CUDA_VISIBLE_DEVICES=7 python dir_main_oi_compact_speedup.py --dataset ogbl_citation2 --split_ratio 0.02\
            --h 15 --tp 16 --c 2 --neg 1 --gamma 3 --pos_ratio 1 --alpha 3 --epoch 100 --batch_edge 16384 \
            --seed 0 --chunks 16 --convergence 1.0 --eval_step 5 \
