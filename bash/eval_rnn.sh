#!/usr/bin/sh

python3 eval.py \
--tag "Thu_Apr_24_22:56:58_2025" \
--model "rnn" \
--window 5 \
--epoch-num 128 \
--batch-size 256 \
--block-size -1 \
--lr 0.0001