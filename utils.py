import os
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from itertools import product
import math
import copy
import time
from sklearn.metrics import confusion_matrix

# we've changed to a faster solver
#from scipy.optimize import linear_sum_assignment
import logging

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
#from datasets import MNIST_truncated, CIFAR10_truncated, ImageFolderTruncated, CIFAR10ColorGrayScaleTruncated
#from combine_nets import prepare_uniform_weights, prepare_sanity_weights, prepare_weight_matrix, normalize_weights, get_weighted_average_pred

#from vgg import *
#from model import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)



def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def parse_class_dist(net_class_config):

    cls_net_map = {}

    for net_idx, net_classes in enumerate(net_class_config):
        for net_cls in net_classes:
            if net_cls not in cls_net_map:
                cls_net_map[net_cls] = []
            cls_net_map[net_cls].append(net_idx)

    return cls_net_map


def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_nets, alpha, args):

    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'cinic10':
        _train_dir = './data/cinic10/cinic-10-trainlarge/train'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        trainset = ImageFolderTruncated(_train_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), 
                                                                                            requires_grad=False),
                                                                                            (4,4,4,4),mode='reflect').data.squeeze()),
                                                            transforms.ToPILImage(),
                                                            transforms.RandomCrop(32),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=cinic_mean,std=cinic_std),
                                                            ]))
        y_train = trainset.get_train_labels
        n_train = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fbs":
        # in this part we conduct a experimental study on exploring the effect of increasing the number of batches
        # but the number of data points are approximately fixed for each batch
        # the arguments we need to use here are: `args.partition_step_size`, `args.local_points`, `args.partition_step`(step can be {0, 1, ..., args.partition_step_size - 1}).
        # Note that `args.partition` need to be fixed as "hetero-fbs" where fbs means fixed batch size
        net_dataidx_map = {}

        # stage 1st: homo partition
        idxs = np.random.permutation(n_train)
        total_num_batches = int(n_train/args.local_points) # e.g. currently we have 180k, we want each local batch has 5k data points the `total_num_batches` becomes 36
        step_batch_idxs = np.array_split(idxs, args.partition_step_size)
        
        sub_partition_size = int(total_num_batches / args.partition_step_size) # e.g. for `total_num_batches` at 36 and `args.partition_step_size` at 6, we have `sub_partition_size` at 6

        # stage 2nd: hetero partition
        n_batches = (args.partition_step + 1) * sub_partition_size
        min_size = 0
        K = 10

        #N = len(step_batch_idxs[args.step])
        baseline_indices = np.concatenate([step_batch_idxs[i] for i in range(args.partition_step + 1)])
        y_train = y_train[baseline_indices]
        N = y_train.shape[0]

        while min_size < 10:
            idx_batch = [[] for _ in range(n_batches)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_batches))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_batches) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        # we leave this to the end
        for j in range(n_batches):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
        return y_train, net_dataidx_map, traindata_cls_counts, baseline_indices

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)

    #return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)
    return y_train, net_dataidx_map, traindata_cls_counts


if __name__ == "__main__":
    print("test utils")