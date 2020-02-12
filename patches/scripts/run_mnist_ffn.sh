#!/bin/bash
for i in {1..100}
do
    echo $i
    python3 clever_wolf_main.py --netname ../nets/mnist_relu_9_200.tf --epsilon 0.045 --dataset mnist --max_cuts 30 --image_num $i --seed 42 2>'err.txt' >"ffn_9_200_$i.txt"
done


