#!/bin/bash
seeds=(0 1 2 3 4)
gpu=0

for seed in ${seeds[@]}
do
    python test.py --test mvtec --gpu $gpu --seed $seed --test_set_path ../dataset/mvtec --save_path results/mvtec/shot-0 --load_path weights/visa_pretrained.pth
    python test.py --test visa --gpu $gpu --seed $seed --test_set_path ../dataset/visa --save_path results/visa/shot-0 --load_path weights/mvtec_pretrained.pth
done