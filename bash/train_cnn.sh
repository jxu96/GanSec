#!/usr/bin/sh

python3 train.py \
--model cnn \
--window 5 \
--ec-epoch-num 512 \
--ec-batch-size 128 \
--ec-block-size -1 \
--ec-lr-g 0.00008 \
--ec-lr-d 0.00006 \
--co-epoch-num 512 \
--co-batch-size 128 \
--co-block-size -1 \
--co-lr-g 0.00008 \
--co-lr-d 0.00004