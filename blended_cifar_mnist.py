import torch
import torchvision
import numpy as np
import timm
from torchvision import transforms
import tqdm
import time
import torch.nn.functional as F
import pdb

class blendedCIFARMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None,randomized=False,correlation_strength=0.9):
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
        # print(len(self.cifar10_dataset),len(self.mnist_dataset))
        self.correlation_strength = correlation_strength
        self.mapper_dict = self.make_cifar_mnist_dict()
        self.transform = transform
        self.pad_mnist = transforms.Pad(2)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((224,224))

    def make_blended(self,cifar_image, mnist_image):
        # select random location to paste
        # select random size of digit
        # paste the images

        offset_x = np.random.randint(10)
        offset_y = np.random.randint(10)

        resize_val = np.random.randint(10,28)

        mnist_image = transforms.functional.resize(mnist_image, resize_val)
        mnist_image = 0.8 * transforms.functional.to_tensor(mnist_image)
        mnist_image = transforms.functional.to_pil_image(mnist_image)

        cifar_image.paste(mnist_image, (offset_x,offset_y), mnist_image)

        return cifar_image


    def make_cifar_mnist_dict(self):
        mapper_dict = {}
        reassignment_amount = 1-self.correlation_strength

        if self.randomized:
            #breaks the easy correlation 
            for k,v in zip(range(len(self.cifar10_dataset)),range(len(self.mnist_dataset))):
                mapper_dict[k] = v
        else:
            sub_c_list = []
            for c in range(10):
                c_l = np.where(np.array(self.cifar10_dataset.targets) == c)[0]
                m_l = np.where(self.mnist_dataset.targets == c)[0]
                if len(m_l) < len(c_l):
                    m_l = np.concatenate([m_l,m_l])
                if self.correlation_strength != 1.0:
                    sub_c_list.append(c_l[:int(reassignment_amount * len(c_l))])
                for k,v in zip(c_l,m_l):
                    mapper_dict[k] = v
            """
            Controlling the amount of correlation.
            Note: this code will be replaced with something more efficient later. 
            right now, priority is to carefully control the spurious correlation.
            """
            reassignment_amount_per_class = -1
            if self.correlation_strength != 1.0:
                reassigned_mnist_indices = []
                reassignment_amount_per_class = int((reassignment_amount * len(c_l)) / 10)
                for enum, sub_c in enumerate(sub_c_list):
                    remap_x = []
                    for x_enum,x_sub_c in enumerate(sub_c_list):
                        if enum != x_enum: 
                            if x_enum < enum:
                                #intentional. shifting.
                                idx_start, idx_end = int((enum-1)*reassignment_amount_per_class),int(((enum-1)+1)*reassignment_amount_per_class)
                            else:
                                idx_start, idx_end = int((enum)*reassignment_amount_per_class),int(((enum)+1)*reassignment_amount_per_class)

                            remap_x.extend([mapper_dict[k] for k in x_sub_c[idx_start:idx_end]])
                    reassigned_mnist_indices.append(remap_x) 
                for orig_l, mnist_l in zip(sub_c_list, reassigned_mnist_indices):
                    for orig_idx, mnist_idx in zip(orig_l,mnist_l):
                        mapper_dict[orig_idx] = mnist_idx
        """
        Count Disagreements to ensure that 
        the appropriate amount of correlation is broken.
        """
        count_disagrees = 0
        for k,v in mapper_dict.items():
            _, label = self.cifar10_dataset[k]
            #get a MNIST sample that shares the class
            mnist_idx = mapper_dict[k]

            #get MNIST part
            _, mlabel = self.mnist_dataset[mnist_idx]

            if mlabel != label:
                count_disagrees += 1

        print() 
        print("Reassignment Amount -- {0:.3f} -- Per Class -- {1}".format(reassignment_amount,reassignment_amount_per_class))
        print("Percentage Disagreements:{0:.3f}".format(count_disagrees/len(mapper_dict.items())))
        print() 
        return mapper_dict 

    def __len__(self):
        return len(self.cifar10_dataset)
    def __getitem__(self,idx):
        sample, label = self.cifar10_dataset[idx]
        #get a MNIST sample that shares the class
        mnist_idx = self.mapper_dict[idx]

        #get MNIST part
        msample, mlabel = self.mnist_dataset[mnist_idx]

        #blend samples        
        new_sample = self.make_blended(sample.copy(),msample )
        new_sample = self.resize(new_sample)
        new_sample = self.to_tensor(new_sample)
        new_sample = self.normalize(new_sample)

        return new_sample,label

class blendedSTLMNIST(torch.utils.data.Dataset):
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
        # print("*** TRAIN DATASET: ",len(train_dataset))
        stl_to_cifar_indices = np.array([0, 2, 1, 3, 4, 5, 7, -1, 8, 9])
        train_dataset.labels = stl_to_cifar_indices[train_dataset.labels]
        self.stl_dataset = torch.utils.data.Subset(train_dataset,np.where(train_dataset.labels != -1)[0])
        self.mnist_dataset = torchvision.datasets.MNIST(root="/p/lustre1/trivedi1/vision_data", 
                                                        train=mnist_train_bool,
                                                        download=True,
                                                        transform = None)
        self.randomized = randomized
        # print(len(self.stl_dataset),len(self.mnist_dataset))
        self.mapper_dict = self.make_cifar_mnist_dict()
        self.transform = transform
        self.pad_mnist = transforms.Pad(2)
        self.resize = transforms.Resize((224,224))
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

    def make_blended(self,cifar_image, mnist_image):
        # select random location to paste
        # select random size of digit
        # paste the images

        offset_x = np.random.randint(10)
        offset_y = np.random.randint(10)

        resize_val = np.random.randint(10,28)

        mnist_image = transforms.functional.resize(mnist_image, resize_val)
        mnist_image = 0.8 * transforms.functional.to_tensor(mnist_image)
        mnist_image = transforms.functional.to_pil_image(mnist_image)

        cifar_image.paste(mnist_image, (offset_x,offset_y), mnist_image)

        return cifar_image

    def __len__(self):
        return len(self.stl_dataset)
    def __getitem__(self,idx):
        sample, label = self.stl_dataset[idx]
        #get a MNIST sample that shares the class
        subset_idx = self.stl_dataset.indices[idx]
        mnist_idx = self.mapper_dict[subset_idx]

        #get MNIST part
        msample, mlabel = self.mnist_dataset[mnist_idx]

        #blend samples        
        new_sample = self.make_blended(sample.copy(),msample )
        new_sample = self.resize(new_sample)
        new_sample = self.to_tensor(new_sample)
        new_sample = self.normalize(new_sample)

        return new_sample,label
