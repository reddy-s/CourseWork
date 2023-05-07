#!/bin/bash

python main.py \
-s veri \
-t veri \
-a resnet50_fc512 \
--root /Users/sangramreddy/Documents/dev/pocs \
--height 224 \
--width 224 \
--gpu-devices 8 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 30 \
--stepsize 10 20 \
--train-batch-size 256 \
--test-batch-size 100 \
--save-dir logs/resnet50_fc512
