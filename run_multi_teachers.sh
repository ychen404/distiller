# Ongoing experiments (4/23)
#  python3 my_main.py --epochs 10 --cloud_epochs 10 --communication_round 10 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 8 --edge_update no_update \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_8_eps_10_cloud_eps_10_comm_10_e_r8_c_r8_frac_1_edge_no_update_constantlr \
#  --log debug --gpu_id 2,3

#  python3 my_main.py --epochs 10 --cloud_epochs 10 --communication_round 10 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 8 --edge_update edge_distillation \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_8_eps_10_cloud_eps_10_comm_10_e_r8_c_r8_frac_1_edge_edge_distillation_constantlr \
#  --log debug --gpu_id 2,3

#  python3 my_main.py --epochs 10 --cloud_epochs 10 --communication_round 10 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 8 --edge_update no_update \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_8_eps_10_cloud_eps_10_comm_10_e_r8_c_r8_frac_1_fedavg_no_update_constantlr \
#  --log debug --gpu_id 2,3

#  python3 my_main.py --epochs 10 --cloud_epochs 10 --communication_round 10 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 8 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_8_eps_10_cloud_eps_10_comm_10_e_r8_c_r8_frac_1_fedavg_copy_constantlr \
#  --log debug --gpu_id 2,3

# # remember to change the batch size back to 128 later
#  python3 my_main.py --epochs 10 --cloud_epochs 10 --communication_round 10 --edge cnn --batch-size 512 \
#  --cloud cnn --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 1 --edge_update no_update \
#  --workspace debug_lr_issue \
#  --log debug --gpu_id 0,1

# 4/24 ongoing follow the federated learning paper
# The changes: epochs is set to 5, batch size is set to 50, and communication round is set to 100 
#  python3 my_main.py --epochs 5 --cloud_epochs 1 --communication_round 100 --edge resnet8 --batch-size 50 \
#  --cloud resnet8 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 8 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_8_eps_5_cloud_eps_1_comm_100_e_r8_c_r8_frac_1_fedavg_copy_constantlr \
#  --log debug --gpu_id 2

# also running 250 communication round, it seems that the accuracy is already plateued 
#   python3 my_main.py --epochs 5 --cloud_epochs 1 --communication_round 250 --edge resnet8 --batch-size 50 \
#  --cloud resnet8 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 8 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_8_eps_5_cloud_eps_1_comm_250_e_r8_c_r8_frac_1_fedavg_copy_constantlr \
#  --log debug --gpu_id 1

#  python3 my_main.py --epochs 10 --cloud_epochs 10 --communication_round 10 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 8 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_8_eps_10_cloud_eps_10_comm_10_e_r8_c_r8_frac_1_multiteacher_kd_copy_constantlr \
#  --log debug --gpu_id 2,3


# 4/26 ongoing test
 # larger batch size test 
 # stick with 10 local eps and 10 communication rounds
#   python3 my_main.py --epochs 10 --cloud_epochs 10 --communication_round 10 --edge resnet8 --batch-size 1024 \
#  --cloud resnet8 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 8 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_8_eps_10_cloud_eps_10_comm_10_e_r8_c_r8_frac_1_fedavg_copy_constantlr_bs_1024 \
#  --log debug --gpu_id 2,3

#   python3 my_main.py --epochs 10 --cloud_epochs 10 --communication_round 10 --edge resnet8 --batch-size 1024 \
#  --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 8 --edge_update no_update \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_8_eps_10_cloud_eps_10_comm_10_e_r8_c_r8_frac_1_multiteacher_kd_copy_constantlr_bs_1024 \
#  --log debug --gpu_id 2


# train for longer time 10 local eps 100 communication_round
#    python3 my_main.py --epochs 10 --cloud_epochs 5 --communication_round 100 --edge resnet8 --batch-size 1024 \
#  --cloud resnet8 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 8 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_8_eps_10_cloud_eps_5_comm_100_e_r8_c_r8_frac_1_fedavg_copy_constantlr_bs_1024 \
#  --log debug --gpu_id 0

# Does FedAvg work better with small local epochs? 
# How about change local epoch back to 10?
#    python3 my_main.py --epochs 10 --cloud_epochs 1 --communication_round 100 --edge resnet8 --batch-size 50 \
#  --cloud resnet8 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 8 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_8_eps_10_cloud_eps_5_comm_100_e_r8_c_r8_frac_1_fedavg_copy_constantlr \
#  --log debug --gpu_id 0

# 4/27 ongoing experiments
#    python3 my_main.py --epochs 5 --cloud_epochs 1 --communication_round 250 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 8 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_20_eps_5_cloud_eps_1_comm_250_e_r8_c_r8_frac_1_fedavg_copy_constantlr_bs_128 \
#  --log debug --gpu_id 0

#    python3 my_main.py --epochs 5 --cloud_epochs 1 --communication_round 250 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode fedavg --frac 1 \
#  --num_users 20 --edge_update no_update \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_20_eps_5_cloud_eps_1_comm_250_e_r8_c_r8_frac_1_fedavg_no_update_constantlr_bs_128 \
#  --log debug --gpu_id 1

#     python3 my_main.py --epochs 5 --cloud_epochs 1 --communication_round 250 --edge resnet8 --batch-size 128 \
#  --cloud resnet8 --dataset cifar10 --mode fedavg --frac 0.4 \
#  --num_users 20 --edge_update copy \
#  --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
#  --workspace n_users_20_eps_5_cloud_eps_1_comm_250_e_r8_c_r8_frac_0.4_fedavg_copy_constantlr_bs_128 \
#  --log debug --gpu_id 2

# debug model copy of FedDF
   python3 my_main.py --epochs 5 --cloud_epochs 1 --communication_round 2 --edge resnet8 --batch-size 128 \
 --cloud resnet8 --dataset cifar10 --mode multiteacher_kd --frac 1 \
 --num_users 8 --edge_update copy \
 --scheduler constant --weight-decay 0 --momentum 0 --no_nesterov \
 --workspace debug_update \
 --log debug --gpu_id 0