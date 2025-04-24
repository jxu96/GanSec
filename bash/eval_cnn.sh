#!/usr/bin/sh

python3 eval.py \
--tag "Wed_Apr_23_09:42:50_2025_CNN" \
--model "cnn" \
--window 5 \
--epoch-num 128 \
--batch-size 256 \
--block-size 32 \
--lr-clf 0.001