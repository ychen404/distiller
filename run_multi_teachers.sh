# Can the fraction causing problem with the alpha?
# Can the checkpoint file from the early stopping causing problem?
# python3 my_main.py --epochs 20 --cloud_epochs 1 --communication_round 40 --edge resnet20 --batch-size 20 \
#  --cloud resnet20 --dataset cifar10 --mode FedAvg --frac 1 --learning_rate 0.01 --optimizer sgd \
#  --cloud_optimizer adam \
#  --num_users 10 --edge_update copy --temperature 1 --cloud_temperature 1 --patience 3 --alpha 0.1\
#  --scheduler constant --weight-decay 5e-4 --momentum 0.9 --no_nesterov --cloud_learning_rate 0.001 \
#  --cloud_scheduler cosineannealinglr \
#  --log debug --workspace FedBE --gpu_id 0

 python3 my_main.py --epochs 10 --cloud_epochs 1 --communication_round 10 --edge resnet8 --batch-size 128 \
 --cloud resnet8 --dataset cifar10 --mode FedAvg --frac 1 --learning_rate 0.1 --optimizer sgd \
 --cloud_optimizer adam \
 --num_users 5 --edge_update copy --temperature 1 --cloud_temperature 1 --patience 3 --alpha 0.01\
 --scheduler constant --weight-decay 5e-4 --momentum 0.9 --no_nesterov --cloud_learning_rate 0.001 \
 --cloud_scheduler cosineannealinglr \
 --log info --workspace FedAvg_alpha_test --gpu_id 0