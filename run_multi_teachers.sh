#    python3 my_main.py --epochs 2 --cloud_epochs 2 --communication_round 2 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 2 --edge_update copy \
#  --scheduler constant --cloud_scheduler cosineannealinglr --weight-decay 0 --momentum 0 --no_nesterov --cloud_learning_rate 0.001 \
#  --workspace test_lr \
#  --log debug --gpu_id 1

# 4/30 ongoing test

#    python3 my_main.py --epochs 5 --cloud_epochs 1 --communication_round 250 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 8 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace fedavg_cloud_model \
#  --log debug --gpu_id 0

#    python3 my_main.py --epochs 5 --cloud_epochs 1 --communication_round 250 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 8 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace fedavg_cloud_model_noupdate \
#  --log debug --gpu_id 0

# 4/31 test
#    python3 my_main.py --epochs 5 --cloud_epochs 1 --communication_round 250 --edge resnet8 --batch-size 50 \
#  --cloud resnet8 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 8 --edge_update no_update \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace fedavg_cloud_model \
#  --log debug --gpu_id 0

# 5/1 test
# edge distillation without label
#    python3 my_main.py --epochs 10 --cloud_epochs 10 --communication_round 250 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 8 --edge_update edge_distillation \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov --ed_nolabel --cloud_learning_rate 0.001 \
#  --cloud_scheduler constant --workspace ed_nolabel \
#  --log debug --gpu_id 1


# 5/4 ongoing test
# edge distillation without label
#    python3 my_main.py --epochs 5 --cloud_epochs 5 --communication_round 250 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 8 --edge_update no_update \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov --cloud_learning_rate 0.001 \
#  --cloud_scheduler constant --workspace fedDF \
#  --log debug --gpu_id 1

#     python3 my_main.py --epochs 5 --cloud_epochs 5 --communication_round 250 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 8 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov --cloud_learning_rate 0.001 \
#  --cloud_scheduler constant --workspace fedDF \
#  --log debug --gpu_id 0

#     python3 my_main.py --epochs 5 --cloud_epochs 5 --communication_round 250 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 8 --edge_update edge_distillation \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov --cloud_learning_rate 0.001 \
#  --cloud_scheduler constant --workspace fedDF \
#  --log debug --gpu_id 2

# Use 100 round instead of 250, because this is can be shown tomorrow 
#    python3 my_main.py --epochs 5 --cloud_epochs 1 --communication_round 100 --edge vgg9 --batch-size 128 \
#  --cloud vgg9 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 8 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace fedavg_vgg \
#  --log debug --gpu_id 3

#     python3 my_main.py --epochs 5 --cloud_epochs 1 --communication_round 100 --edge vgg9 --batch-size 128 \
#  --cloud vgg9 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 8 --edge_update no_update \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace fedavg_vgg \
#  --log debug --gpu_id 3

# 5/6 ongoing experiment
#    python3 my_main.py --epochs 40 --cloud_epochs 5 --communication_round 250 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 8 --edge_update no_update \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov --cloud_learning_rate 0.001 \
#  --cloud_scheduler constant --workspace fedDF \
#  --log debug --gpu_id 1

#   python3 my_main.py --epochs 40 --cloud_epochs 5 --communication_round 250 --edge resnet8 --batch-size 128 \
# --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
# --num_users 8 --edge_update copy \
# --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov --cloud_learning_rate 0.001 \
# --cloud_scheduler constant --workspace fedDF \
# --log debug --gpu_id 0

#     python3 my_main.py --epochs 40 --cloud_epochs 40 --communication_round 250 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 8 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov --cloud_learning_rate 0.001 \
#  --cloud_scheduler constant --workspace fedDF \
#  --log debug --gpu_id 1

#     python3 my_main.py --epochs 10 --cloud_epochs 10 --communication_round 100 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 8 --edge_update copy --temperature 1 --cloud_temperature 1 \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov --cloud_learning_rate 0.001 \
#  --cloud_scheduler constant --workspace fedDF_T_1 \
#  --log debug --gpu_id 0,1

# 5/11
# Test bigger batch size
#     python3 my_main.py --epochs 10 --cloud_epochs 10 --communication_round 100 --edge resnet8 --batch-size 512 \
#  --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 --learning_rate 0.04 \
#  --num_users 8 --edge_update copy --temperature 1 --cloud_temperature 1 \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov --cloud_learning_rate 0.001 \
#  --cloud_scheduler constant --workspace test_speed \
#  --log debug --gpu_id 0,1

#     python3 my_main.py --epochs 10 --cloud_epochs 10 --communication_round 100 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode FedDF --frac 1 --learning_rate 0.1 --cloud_optimizer adam\
#  --num_users 8 --edge_update copy --temperature 1 --cloud_temperature 1 \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov --cloud_learning_rate 0.001 \
#  --cloud_scheduler cosineannealinglr --workspace FedDF_adam \
#  --log debug --gpu_id 2,3

     python3 my_main.py --epochs 3 --cloud_epochs 1 --communication_round 3 --edge resnet8 --batch-size 128 \
 --cloud resnet8 --dataset cifar10 --mode FedDF --frac 1 --learning_rate 0.1 --cloud_optimizer adam\
 --num_users 4 --edge_update copy --temperature 1 --cloud_temperature 1 \
 --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov --cloud_learning_rate 0.001 \
 --cloud_scheduler cosineannealinglr --workspace test_loader \
 --log debug --gpu_id 2,3