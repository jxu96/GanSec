#!/usr/bin/sh

python3 main.py \
--window 5 \
--epoch-num 32 \
--batch-size 64 \
--block-size 4 \
--lr-clf 0.001 \
--epoch-num-gan 10000 \
--batch-size-gan 64 \
--block-size-gan 100 \
--lr-g 0.0015 \
--lr-d 0.0008
