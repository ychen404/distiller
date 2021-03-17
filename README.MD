## Introduction
This repository contains the code for heterogenous federated distillation, which leverage knowledge distillation to help federated learning support heterogenous client architectures.

## Python Dependencies
This codebase only supports Python 3.6+.

Required Python packages:
- `torch torchvision tqdm numpy pandas seaborn`

All packages can be installed using `pip3 install --user -r requirements.txt`.

## Run
You can run the code with various settings, for example:
python3 my_main.py --epochs 10 --communication_round 100 --teacher resnet8 --student resnet8 --dataset cifar10 --mode multiteacher_kd --num_users 8

