#  python3 my_main.py --epochs 1 --communication_round 1 --teacher resnet8 --batch-size 1024 \
#  --student resnet8 --dataset cifar10 --mode multiteacher_kd \
#  --num_users 1 --edge_distillation 0 --workspace test_refactor_names --teacher-checkpoint 'pretrained/resnet8_cifar10.pth'


 python3 my_main.py --epochs 10 --communication_round 10 --teacher resnet8 --batch-size 1024 \
 --student resnet8 --dataset cifar10 --mode multiteacher_kd \
 --num_users 2 --edge_distillation 1 --workspace num_users_2_epochs_2_comm_10_edge_distil_1