#!/usr/bin/sh

python3 main.py \
--window 5 \
--epoch-num 100 \
--batch-size 128 \
--block-size 10 \
--lr-clf 0.001 \
--epoch-num-gan 2048 \
--batch-size-gan 128 \
--block-size-gan 256 \
--lr-g 0.0015 \
--lr-d 0.0008
