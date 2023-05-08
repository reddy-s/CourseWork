# Course Work EEEM071 (2023 Spring)

gdown 1TsQUh4i5JeUuCaFhCSXaUq0HN9f5c28s

nvidia-docker run -it --rm -v /home/ec2-user:/workspace pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel sh

cd CourseWork

python3 main.py -s veri -t veri -a resnet50 --root /home/ec2-user --height 224 --width 224 --optim amsgrad --lr 0.0003 --max-epoch 1 --stepsize 10 20 --train-batch-size 256 --gpu-devices 4 --test-batch-size 100 --save-dir logs/resnet50

aws s3 cp --recursive ./logs/resnet50_fc512 s3://ml-workflow/models/resnet50_fc512_i44 && rm -rf logs/*
