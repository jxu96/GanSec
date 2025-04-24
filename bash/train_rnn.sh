#!/usr/bin/sh

python3 train.py \
--model rnn \
--window 5 \
--ec-epoch-num 500 \
--ec-batch-size 64 \
--ec-block-size -1 \
--ec-lr-g 0.0001 \
--ec-lr-d 0.0001 \
--co-epoch-num 500 \
--co-batch-size 64 \
--co-block-size -1 \
--co-lr-g 0.0001 \
--co-lr-d 0.0001