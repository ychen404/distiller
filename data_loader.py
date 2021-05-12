from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from utils.timer import Timer
from torch.utils.data.sampler import SubsetRandomSampler


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
def get_cifar(num_classes=100, dataset_dir="./data", batch_size=128, split=0.1,
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

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               num_workers=NUM_WORKERS,
                                               sampler=train_sampler,
                                               pin_memory=True, shuffle=False)

    valid_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=batch_size,
                                            num_workers=NUM_WORKERS,
                                            sampler=valid_sampler,
                                            pin_memory=True, shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              num_workers=NUM_WORKERS,
                                              pin_memory=True, shuffle=False)


    return train_loader, test_loader, valid_loader, train_data, test_data
