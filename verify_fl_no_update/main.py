import argparse
import torch.nn as nn
import torch
import os, sys
# import aggregate_method.Fed
# import util
from resnet import ResNet8
import pkgutil
from collections import OrderedDict
import copy
from math import pi
from math import cos
from math import floor
import torchvision
import torchvision.transforms as transforms
import time


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def load_checkpoint(model, checkpoint_path, device='cpu'):
    device = torch.device(device)
    model_ckp = torch.load(checkpoint_path, map_location=device)

    # handle both dataparallel and normal models
    model_tmp_dict = OrderedDict()
    for name, value in model_ckp["model_state_dict"].items():
        if name.startswith("module."):
            name = name[7:]
        model_tmp_dict[name] = value

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(model_tmp_dict)
    else:
        model.load_state_dict(model_tmp_dict)
    return model


last_time = time.time()
begin_time = last_time
TOTAL_BAR_LENGTH = 65.


def test(net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f"Test Accuracy: {100.*correct/total}")

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--net', default='res8')
parser.add_argument('--bs', default='128')
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

bs = int(args.bs)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

num_users = 8

# ckpts = {'0':'resnet8_edge_0_ckpt.pth.tar',
# '1':'resnet8_edge_1_ckpt.pth.tar',
# '2':'resnet8_edge_2_ckpt.pth.tar',
# '3':'resnet8_edge_3_ckpt.pth.tar',
# '4':'resnet8_edge_4_ckpt.pth.tar',
# '5':'resnet8_edge_5_ckpt.pth.tar',
# '6':'resnet8_edge_6_ckpt.pth.tar',
# '7':'resnet8_edge_7_ckpt.pth.tar'
# }

ckpts = {'0':'with_update_249/resnet8_edge_0_ckpt.pth.tar',
'1':'with_update_249/resnet8_edge_1_ckpt.pth.tar',
'2':'with_update_249/resnet8_edge_2_ckpt.pth.tar',
'3':'with_update_249/resnet8_edge_3_ckpt.pth.tar',
'4':'with_update_249/resnet8_edge_4_ckpt.pth.tar',
'5':'with_update_249/resnet8_edge_5_ckpt.pth.tar',
'6':'with_update_249/resnet8_edge_6_ckpt.pth.tar',
'7':'with_update_249/resnet8_edge_7_ckpt.pth.tar'
}




# init models
edge_nets = []
print('==> Building edge models..')
for i in range(num_users):
    edge_net = ResNet8()
    edge_nets.append(edge_net)

print('==> Building cloud model..')
cloud_net = ResNet8()

# load all the edge models
for i, net in enumerate(edge_nets):
    path = ckpts[str(i)]
    print(f"Loading model from {path}")
    net = load_checkpoint(net, checkpoint_path=path)

all_edge_weights = [None for i in range(num_users)]
print(all_edge_weights)

# for user, partition_idx in enumerate (idxs_users):
#                 w = edge_nets[user].state_dict()
#                 all_edge_weights[user] = copy.deepcopy(w)               
#             averaged_weights = FedAvg(all_edge_weights)

# calculate the averaged model 
for i, net in enumerate(edge_nets):
    w = net.state_dict()
    all_edge_weights[i] = copy.deepcopy(w)

averaged_weights = FedAvg(all_edge_weights)

# check the last layer weights and they are indeed different
for w in all_edge_weights:
    print(w['linear.bias'])

print(f"averaged_weights: {averaged_weights['linear.bias']}")

# calculate accuracy
cloud_net.load_state_dict(averaged_weights)
for run in range(0, 3):
    test(cloud_net)