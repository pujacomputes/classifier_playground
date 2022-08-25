import torch
import torchvision
import numpy as np
import timm
from torchvision import transforms
import tqdm
import time
import torch.nn.functional as F
import pdb
class pairedCIFARMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None,randomized=False):
        if train:
            cifar_train_bool = True
            mnist_train_bool = True
        else:
            cifar_train_bool = False 
            mnist_train_bool = False 

        self.cifar10_dataset = torchvision.datasets.CIFAR10(root="/p/lustre1/trivedi1/vision_data",train=cifar_train_bool)
                

        self.mnist_dataset = torchvision.datasets.MNIST(root="/p/lustre1/trivedi1/vision_data", 
                                                        train=mnist_train_bool,
                                                        download=True,
                                                        transform = None)
        self.randomized = randomized
        print(len(self.cifar10_dataset),len(self.mnist_dataset))
        self.mapper_dict = self.make_cifar_mnist_dict()
        self.transform = transform
        self.pad_mnist = transforms.Pad(2)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
    
    def make_cifar_mnist_dict(self):
        mapper_dict = {}
        if self.randomized:
            #breaks the easy correlation 
            for k,v in zip(range(len(self.cifar10_dataset)),range(len(self.mnist_dataset))):
                mapper_dict[k] = v
        else: 
            for c in range(10):
                c_l = np.where(np.array(self.cifar10_dataset.targets) == c)[0]
                m_l = np.where(self.mnist_dataset.targets == c)[0]
                if len(m_l) < len(c_l):
                    m_l = np.concatenate([m_l,m_l])
                for k,v in zip(c_l,m_l):
                    mapper_dict[k] = v
        return mapper_dict 

    def __len__(self):
        return len(self.cifar10_dataset)
    def __getitem__(self,idx):
        sample, label = self.cifar10_dataset[idx]
        #get a MNIST sample that shares the class
        mnist_idx = self.mapper_dict[idx]

        #get MNIST part
        msample, mlabel = self.mnist_dataset[mnist_idx]
        msample = self.pad_mnist(msample)
        msample = self.to_tensor(msample)
        msample = msample.repeat(3,1,1)
        sample = self.to_tensor(sample)

        new_sample = torch.cat([sample,msample],dim=1)
        new_sample = self.normalize(new_sample)
        return new_sample,label

class pairedSTLMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None,randomized=False):
        if train:
            stl_train_bool = "train"
            mnist_train_bool = True
        else:
            stl_train_bool = "test"
            mnist_train_bool = False 

        train_dataset = torchvision.datasets.STL10(root="/p/lustre1/trivedi1/vision_data",
            split=stl_train_bool,
            download=False,
            transform=None)
        print("*** TRAIN DATASET: ",len(train_dataset))
        stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
        train_dataset.labels = stl_to_cifar_indices[train_dataset.labels]
        self.stl_dataset = torch.utils.data.Subset(train_dataset,np.where(train_dataset.labels != -1)[0])
        self.mnist_dataset = torchvision.datasets.MNIST(root="/p/lustre1/trivedi1/vision_data", 
                                                        train=mnist_train_bool,
                                                        download=True,
                                                        transform = None)
        self.randomized = randomized
        print(len(self.stl_dataset),len(self.mnist_dataset))
        self.mapper_dict = self.make_cifar_mnist_dict()
        self.transform = transform
        self.pad_mnist = transforms.Pad(2)
        self.resize = transforms.Resize((32,32))
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
    def make_cifar_mnist_dict(self):
        mapper_dict = {}
        if self.randomized:
            #breaks the easy correlation 
            for k,v in zip(range(len(self.stl_dataset.dataset.labels)),range(len(self.mnist_dataset))):
                mapper_dict[k] = v
        else: 
            for c in range(10):
                c_l = np.where(np.array(self.stl_dataset.dataset.labels) == c)[0]
                m_l = np.where(self.mnist_dataset.targets == c)[0]
                if len(m_l) < len(c_l):
                    m_l = np.concatenate([m_l,m_l])
                for k,v in zip(c_l,m_l):
                    mapper_dict[k] = v
    
        return mapper_dict

    def __len__(self):
        return len(self.stl_dataset)
    def __getitem__(self,idx):
        sample, label = self.stl_dataset[idx]
        #get a MNIST sample that shares the class
        subset_idx = self.stl_dataset.indices[idx]
        mnist_idx = self.mapper_dict[subset_idx]

        #get MNIST part
        msample, mlabel = self.mnist_dataset[mnist_idx]
        msample = self.pad_mnist(msample)
        msample = self.to_tensor(msample)
        msample = msample.repeat(3,1,1)
        sample = self.resize(sample)
        sample = self.to_tensor(sample)

        new_sample = torch.cat([sample,msample],dim=1)
        new_sample = self.normalize(new_sample)
        return new_sample,label
