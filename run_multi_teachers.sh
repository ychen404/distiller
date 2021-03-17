# python3 my_main.py --epochs 10 --teacher resnet8 --student resnet8 --dataset cifar10 --mode kd
python3 my_main.py --epochs 1 --communication_round 1 --teacher resnet8 --student resnet8 --dataset cifar10 --mode multiteacher_kd --num_users 8
