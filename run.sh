#!/usr/bin/sh

python3 main.py \
--window 5 \
--epoch-num 100 \
--batch-size 64 \
--block-size 10 \
--lr-clf 0.00012 \
--epoch-num-gan 1000 \
--batch-size-gan 64 \
--block-size-gan 100 \
--lr-g 0.0001 \
--lr-d 0.00005
