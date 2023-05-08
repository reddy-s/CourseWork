#!/bin/bash

python main.py \
-s veri \
-t veri \
-a resnet18_fc512 \
--root /Users/sangramreddy/Documents/dev/pocs \
--height 224 \
--width 224 \
--gpu-devices 16 \
--optim amsgrad \
--lr 0.003 \
--max-epoch 30 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir logs/resnet18_fc512
