#!/bin/bash
for i in {1..100}
do
    echo $i
    python3 clever_wolf_main.py --netname ../nets/convSmallRELU__Point.pyt --epsilon 0.12 --dataset mnist --image_num $i --seed 42 2>'err.txt' >"convSmall_$i.txt"
done


