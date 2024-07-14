import torch
import os
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
import PIL
import numpy as np


class RandomCrop(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)

class Dataset_train(Dataset):
    def __init__(self, dataset_size, path_pos, path_neg, path_gdt, device):
        super(Dataset_train, self).__init__()
        
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.path_gdt = path_gdt
        self.list_pos = os.listdir(self.path_pos)
        self.list_neg = os.listdir(self.path_neg)
        self.list_gdt = os.listdir(self.path_gdt)
        self.list_pos.sort()
        self.list_neg.sort()
        self.list_gdt.sort()
        self.num_pos = len(self.list_pos)
        self.num_neg = len(self.list_neg)
        self.num_gdt = len(self.list_gdt)
        self.device = device
        self.size = dataset_size
        self.transforms = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),                          
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                       ])
        self.transforms_grdth = transforms.Compose([
            transforms.Resize(self.size)
        ])

    def __getitem__(self, index):
        seed = torch.random.seed()

        if index < self.num_pos:
            image = self.read(self.path_pos, self.list_pos[index], 'train')
            label = torch.ones(1)
            grdth = self.read(self.path_gdt, self.list_gdt[index], 'grdth')
        else:
            image = self.read(self.path_neg, self.list_neg[index - self.num_pos],'train')
            label = torch.zeros(1)
            grdth = torch.zeros(1, self.size[0], self.size[1])

        return image, label, grdth

    def __len__(self):
        return self.num_pos + self.num_neg

    def read(self, path, name, norm=None):
        try:
            img = io.imread(os.path.join(path, name))
        except ValueError:
            print(os.path.join(path, name))
        
        if norm == 'train':
            img = Image.fromarray(img)
            img = self.transforms(img)

        elif norm == 'grdth':
            if len(img.shape) > 2:
                img = np.mean(img,axis=2)
            img = torch.from_numpy(img).float().unsqueeze(0)
            img = self.transforms_grdth(img)
            img = (img > 128) + 0
        return img

class Dataset_valid(Dataset):
    def __init__(self, dataset_size, path_pos, path_neg, path_gdt, device):
        super(Dataset_valid, self).__init__()
            
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.path_gdt = path_gdt
        self.list_pos = os.listdir(self.path_pos)
        self.list_neg = os.listdir(self.path_neg)
        self.list_gdt = os.listdir(self.path_gdt)
        self.list_pos.sort()
        self.list_neg.sort()
        self.list_gdt.sort()
        self.num_pos = len(self.list_pos)
        self.num_neg = len(self.list_neg)
        self.device = device
        self.size = dataset_size
        self.transforms_test = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),                          
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                        ])
        self.transforms_grdth = transforms.Compose([
            transforms.Resize(self.size)
                        ])

    def __getitem__(self, index):

        if index < self.num_pos:
            image = self.read(self.path_pos, self.list_pos[index], 'test')
            grdth = self.read(self.path_gdt, self.list_gdt[index], 'grdth')
        else:
            image = self.read(self.path_neg, self.list_neg[index-self.num_pos], 'test')
            grdth = torch.zeros(1, self.size[0], self.size[1])
        
        return image, grdth

    def __len__(self):
        return self.num_pos + self.num_neg

    def read(self, path, name, norm=None):

        try:
            img = io.imread(os.path.join(path, name))
        except ValueError:
            print(os.path.join(path, name))    

        if norm == 'test':
            img = Image.fromarray(img)
            img = self.transforms_test(img)

        elif norm == 'grdth':
            if len(img.shape) > 2:
                img = np.mean(img,axis=2)
            img = torch.from_numpy(img).float().unsqueeze(0)
            img = self.transforms_grdth(img)
            img = (img > 128) + 0

        return img