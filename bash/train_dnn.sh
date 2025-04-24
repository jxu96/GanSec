#!/usr/bin/sh

python3 train.py \
--model dnn \
--window 5 \
--epoch-num 128 \
--batch-size 256 \
--block-size 64 \
--lr-clf 0.01 \
--epoch-num-gan 20480 \
--batch-size-gan 64 \
--block-size-gan 2560 \
--lr-g 0.015 \
--lr-d 0.008