#!/bin/bash
for i in {1..100}
do
    echo $i
    python3 clever_wolf_main.py --netname ../nets/convSmallRELU__cifar10_Point.pyt --max_cuts 30 --epsilon 0.006 --dataset cifar10 --image_num $i --obox_approx --seed 42 2>'err.txt' >"convSmall_cifar_$i.txt"
done
