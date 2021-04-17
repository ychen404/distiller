# python3 my_main.py --epochs 10 --communication_round 10 --edge resnet8 --batch-size 1024 \
#  --cloud resnet18 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 4 --edge_distillation 0 \
#  --workspace num_users_4_epochs_10_comm_10_e_r8_c_r18_edge_distil_0 --log debug

#  python3 my_main.py --epochs 10 --cloud_epochs 1 --communication_round 10 --edge resnet8 --batch-size 128 \
#  --cloud resnet18 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 20 --edge_distillation 0 \
#  --workspace num_users_20_epochs_10_cloud_eps_1_comm_10_e_r8_c_r18_frac_1_edge_distil_0 --log debug

 python3 my_main.py --epochs 10 --cloud_epochs 1 --communication_round 10 --edge resnet8 --batch-size 128 \
 --cloud resnet18 --dataset cifar10 --mode multiteacher_kd --frac 1 \
 --num_users 20 --edge_distillation 0 \
 --workspace n_users_20_eps_10_cloud_eps_1_comm_10_e_r8_c_r18_frac_1_edge_distil_0 --log debug