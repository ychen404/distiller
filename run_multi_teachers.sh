# python3 my_main.py --epochs 10 --cloud_epochs 10 --communication_round 10 --edge resnet8 --batch-size 128 \
#  --cloud resnet18 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 8 --edge_distillation 0 \
#  --workspace n_users_8_eps_10_cloud_eps_10_comm_10_e_r8_c_r18_frac_1_edge_distil_0 --log debug --gpu_id 2,3

 python3 my_main.py --epochs 1 --cloud_epochs 1 --communication_round 2 --edge resnet8 --batch-size 128 \
 --cloud resnet18 --dataset cifar10 --mode multiteacher_kd --frac 1 \
 --num_users 8 --edge_distillation 1 \
 --workspace test_timer --log debug --gpu_id 2,3