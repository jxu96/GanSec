#!/usr/bin/sh

python3 main.py \
--window 5 \
--epoch-num 10 \
--batch-size 64 \
--block-size 1 \
--lr-clf 0.001 \
--epoch-num-gan 10 \
--batch-size-gan 64 \
--block-size-gan 1 \
--lr-g 0.0002 \
--lr-d 0.0002 \
