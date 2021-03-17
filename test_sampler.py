import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from six.moves import urllib
from data_loader import get_cifar

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


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

class Test(object):
    def __getitem__(self, items):
        print(type(items), items)

class Person:
    def __init__(self,name,age):
        self.name = name
        self.age = age

    def __getitem__(self,key):
        print ("Inside `__getitem__` method!")
        return getattr(self,key)


if __name__ == '__main__':
    dataset_train = datasets.CIFAR10('../data/cifar/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    
    # print(f"dataset_train[0]: {dataset_train[0]}")
    num_users = 100
    idxs_users = cifar_iid(dataset_train, num_users)
    print(idxs_users.keys())
    # print(list(idxs_users[0]))
    print(len(idxs_users[0]))
    


    ldr_train = DataLoader(DatasetSplit(dataset_train, idxs_users), batch_size=128, shuffle=True)
    print(f"The length of ldr_train is {len(ldr_train)}")
    # print(ldr_train[0])
    # for (batch_idx, batch) in enumerate(ldr_train):
    #     print("\nBatch = " + str(batch_idx))
    #     for (x, y) in enumerate(batch):
    #         print(f"\nx = {x}")
    #         print(f"\ny = {y}")