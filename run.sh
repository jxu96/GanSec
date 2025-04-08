#!/usr/bin/sh

python3 main.py \
--window 5 \
--epoch-num 100 \
--batch-size 64 \
--block-size 10 \
--lr-clf 0.001 \
--epoch-num-gan 100 \
--batch-size-gan 64 \
--block-size-gan 10 \
--lr-g 0.0002 \
--lr-d 0.0002 \
