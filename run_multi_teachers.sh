#  python3 my_main.py --epochs 100 --cloud_epochs 100 --communication_round 1 --edge resnet8 --batch-size 128 \
#  --cloud resnet18 --dataset cifar10 --mode multiteacher_kd --frac 1 \
#  --num_users 8 --edge_distillation 0 \
#  --workspace n_users_8_eps_100_cloud_eps_100_comm_1_e_r8_c_r18_frac_1_edge_distil_0 --log debug

python3 my_main.py --epochs 100 --cloud_epochs 100 --communication_round 1 --edge resnet8 --batch-size 128 \
 --cloud resnet18 --dataset cifar10 --mode multiteacher_kd --frac 1 \
 --num_users 20 --edge_distillation 1 \
 --workspace test_edge_distil_1 --log debug --gpu_id 0,1