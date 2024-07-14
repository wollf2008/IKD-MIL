import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
import os
import random
import numpy as np


from models import *
from functions import *
from datasets import Dataset_train, Dataset_valid
import warnings
warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    print('Loading......')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_name = 'camelyon16'
    path_work = 'work/test/'
    if os.path.exists(path_work) is False:
        os.makedirs(path_work)

    path_train_pos = './data/' + dataset_name + '/train/pos'
    path_train_neg = './data/' + dataset_name + '/train/neg'
    path_train_gdt = './data/' + dataset_name + '/train/gt'



    path_valid_pos = './data/' + dataset_name + '/test/pos'
    path_valid_neg = './data/' + dataset_name + '/test/neg'
    path_valid_gdt = './data/' + dataset_name + '/test/gt'


    dataset_size = [256, 256]
    dataset_train = Dataset_train(dataset_size, path_train_pos, path_train_neg, path_train_gdt, device)
    dataset_valid = Dataset_valid(dataset_size, path_valid_pos, path_valid_neg, path_valid_gdt, device)
    
    batch_size = 16
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=0)

    teacher_model = VGG_MIL().to(device)
    student_model = VGG_MIL().to(device)


    hyperparameters = {
        'r' : 4,
        'lr' : 5e-5,
        'wd' : 0.00005,
        'epoch' : 500,
        'pretrain' : True,
        'optimizer' : None
    }

    print('Dataset: ' + dataset_name)
    print('Data Volume: ', len(dataloader_train.dataset))
    print('Teacher Model: ', type(teacher_model))
    print('Student Model: ', type(student_model))
    print('Batch Size: ', batch_size)
    train(path_work, teacher_model, student_model, dataloader_train, device, hyperparameters, valid, dataloader_valid)

if __name__ == '__main__':
    setup_seed(111)
    main()