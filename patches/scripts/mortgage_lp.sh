#!/bin/bash
for file in ../data/mortgage/*.txt
do
    #IMG=${file/\//} 
    #IMG=sed 's/\///g' "$file"
    IMG=$( echo "$file" | sed 's/\/local\/home\/dimitadi\/symadex\/data\/mortgage\/spec//g' )
    IMG=${IMG/.txt/} 
    for SEED in 10 20 30 40 50
    do
        echo "$IMG $SEED" 
        python3 clever_wolf_main.py --netname ../nets/mortgage.pyt --dataset mortgage --image_num $IMG --domain LP --seed $SEED --visualize &> mortgage_${IMG}_${SEED}.txt       
    done
done
