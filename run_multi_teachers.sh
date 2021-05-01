   python3 my_main.py --epochs 2 --cloud_epochs 2 --communication_round 2 --edge resnet8 --batch-size 128 \
 --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
 --num_users 2 --edge_update copy \
 --scheduler constant --cloud_scheduler cosineannealinglr --weight-decay 0 --momentum 0 --no_nesterov --cloud_learning_rate 0.001 \
 --workspace test_lr \
 --log debug --gpu_id 1


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
