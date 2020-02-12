#!/bin/bash
for i in {1..100}
do
echo $i
    python3 clever_wolf_main.py --netname ../nets/ConvBig__Point_mnist.pyt --epsilon 0.05 --max_cuts 30 --dataset mnist --image_num $i --nowolf --seed 42 2>'err.txt' >"convBig_$i.txt"
done
