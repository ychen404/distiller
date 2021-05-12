from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from utils.timer import Timer
from torch.utils.data.sampler import SubsetRandomSampler
import pdb


NUM_WORKERS = 4

class TensorImgSet(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        self.imgs = tensors[0]
        self.targets = tensors[1]
        self.tensors = tensors
        self.transform = transform
        self.len = len(self.imgs)

    def __getitem__(self, index):
        x = self.imgs[index]
        if self.transform:
            x = self.transform(x)
        y = self.targets[index]
        return x, y

    def __len__(self):
        return self.len


# 'Subset' object has no attribute 'targets
class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

def load_cifar_10_1():
    # @article{recht2018cifar10.1,
    #  author = {Benjamin Recht and Rebecca Roelofs and Ludwig Schmidt
    #  and Vaishaal Shankar},
    #  title = {Do CIFAR-10 Classifiers Generalize to CIFAR-10?},
    #  year = {2018},
    #  note = {\url{https://arxiv.org/abs/1806.00451}},
    # }
    # Original Repo: https://github.com/modestyachts/CIFAR-10.1
    data_path = Path(__file__).parent.joinpath("cifar10_1")
    label_filename = data_path.joinpath("v6_labels.npy").resolve()
    imagedata_filename = data_path.joinpath("v6_data.npy").resolve()
    print(f"Loading labels from file {label_filename}")
    labels = np.load(label_filename)
    print(f"Loading image data from file {imagedata_filename}")
    imagedata = np.load(imagedata_filename)
    return imagedata, torch.Tensor(labels).long()


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

@Timer(text='get_cifar in {:.4f} seconds')

def get_cifar(num_classes=100, dataset_dir="./data", batch_size=128, split=0,
              use_cifar_10_1=False):

    if num_classes == 10:
        print("Loading CIFAR10...")
        dataset = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        print("Loading CIFAR100...")
        dataset = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
    ])

    train_data = dataset(root=dataset_dir, train=True,
                       download=True, transform=train_transform)

    test_data = dataset(root=dataset_dir, train=False,
                          download=True,
                          transform=test_transform)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(split * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # pdb.set_trace()

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               num_workers=NUM_WORKERS,
                                               sampler=train_sampler,
                                               pin_memory=True, shuffle=False)

    val_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=batch_size,
                                            num_workers=NUM_WORKERS,
                                            sampler=valid_sampler,
                                            pin_memory=True, shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              num_workers=NUM_WORKERS,
                                              pin_memory=True, shuffle=False)


    return train_loader, test_loader, val_loader, train_data, test_data
    # return train_data, test_data

def split_train_data(train_data, split=0):

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(split * num_train))
    
    train_idx, val_idx = indices[split:], indices[:split]
    train_labels, val_labels = train_data.targets, train_data
    
    # define samplers for obtaining training and validation batches
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(val_idx)
    # train_set = torch.utils.data.Subset(train_data, train_sampler)
    # val_set = torch.utils.data.Subset(train_data, valid_sampler)

    train_set = torch.utils.data.Subset(train_data, train_idx)
    val_set = torch.utils.data.Subset(train_data, val_idx)

    
    return train_set, val_set


## Moving the the below part of loader

def get_loaders(train_data, labels, val_data, test_data, n_clients=10, alpha=0, batch_size=128, n_data=None, num_workers=NUM_WORKERS, seed=0, split=0):

    # Split the labels
    num_train = len(labels)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(split * num_train))
    
    # labels = train_data.targets
    labels_split = labels[split:]

    # subset_idcs = split_dirichlet(train_data.targets, n_clients, n_data, alpha, seed=seed)
    # client_data = [torch.utils.data.Subset(train_data, subset_idcs[i]) for i in range(n_clients)]

    subset_idcs = split_dirichlet(labels_split, n_clients, n_data, alpha, seed=seed)
    # pdb.set_trace()
    client_data = [torch.utils.data.Subset(train_data, subset_idcs[i]) for i in range(n_clients)]


    client_loaders = [torch.utils.data.DataLoader(subset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=num_workers, 
                                                    pin_memory=True) for subset in client_data]

    val_loader = torch.utils.data.DataLoader(val_data, 
                                                batch_size=batch_size,
                                                shuffle=True, 
                                                num_workers=num_workers, 
                                                pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_data, 
                                                batch_size=batch_size, 
                                                num_workers=num_workers, 
                                                pin_memory=True)


    return client_loaders, val_loader, test_loader


def split_dirichlet(labels, n_clients, n_data, alpha, double_stochstic=True, seed=0):
    '''Splits data among the clients according to a dirichlet distribution with parameter alpha'''

    np.random.seed(seed)

    if isinstance(labels, torch.Tensor):
      labels = labels.numpy()

    n_classes = np.max(labels)+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    if double_stochstic:
      label_distribution = make_double_stochstic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels)==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    print_split(client_idcs, labels)
  
    return client_idcs

def make_double_stochstic(x):
    rsum = None
    csum = None

    n = 0 
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    return x

def print_split(idcs, labels):
    n_labels = np.max(labels) + 1 
    print("Data split:")
    splits = []
    for i, idccs in enumerate(idcs):
        split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
        splits += [split]
        if len(idcs) < 30 or i < 10 or i>len(idcs)-10:
            print(" - Client {}: {:55} -> sum={}".format(i,str(split), np.sum(split)), flush=True)
        elif i==len(idcs)-10:
            print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)

    print(" - Total:     {}".format(np.stack(splits, axis=0).sum(axis=0)))
    print()


class IdxSubset(torch.utils.data.Dataset):

    def __init__(self, dataset, indices, return_index):
        self.dataset = dataset
        self.indices = indices
        self.return_index = return_index

    def __getitem__(self, idx):
        if self.return_index:
          return self.dataset[self.indices[idx]], idx
        else:
          return self.dataset[self.indices[idx]]#, idx

    def __len__(self):
        return len(self.indices)