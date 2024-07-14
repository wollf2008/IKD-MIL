import torch
import torch.nn as nn
import math
import torchvision
import cv2
import numpy as np
import torchvision.transforms as transforms
import PIL
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from matplotlib import cm
import torchvision.models as models
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from models import *


class DiceLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (
                torch.sum(output) + torch.sum(target) + self.smooth + self.eps)

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100,
                  custom_str='oeem'):
    """The wrapper function for :func:`F.cross_entropy`"""

    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    label = label.squeeze(1).long()
    inverted_pred = 1 - pred
    pred = torch.cat((inverted_pred, pred), dim=1)
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)
    # online easy example mining 
    if 'oeem' in custom_str:

        # normalized loss
        weight = torch.ones_like(loss)
        metric = -loss.detach().reshape((loss.shape[0], loss.shape[1] * loss.shape[2]))
        weight = F.softmax(metric, 1)
        weight = weight / weight.mean(1).reshape((-1, 1))
        weight = weight.reshape((loss.shape[0], loss.shape[1], loss.shape[2]))
        
        # apply oeem on images of multiple labels
        for i in range(label.shape[0]):
            tag = set(label[i].reshape(label.shape[1] * label.shape[2]).tolist())
            if len(tag) <= 1:
                weight[i] = 1

    # apply weights and reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def train(path_work, teacher_model, student_model, dataloader_train, device, hp, valid_fn=None, dataloader_valid=None, dataloader_update = None):


    if hp['pretrain'] is True:
        teacher_model.pretrain()
        teacher_model = teacher_model.to(device)
        student_model.pretrain()
        student_model = student_model.to(device)

    r = hp['r']
    lr = hp['lr']
    wd = hp['wd']
    num_epoch = hp['epoch']
    best_result = 0
    best_accuracy = 0
    print('Learning Rate: ', lr)
    criterion = torch.nn.CrossEntropyLoss()
    dataset_size = len(dataloader_train.dataset)
    dice = DiceLoss(smooth=1)

    if hp['optimizer'] == 'side':
        params1_teacher = list(map(id, teacher_model.decoder1.parameters()))
        params2_teacher = list(map(id, teacher_model.decoder2.parameters()))
        params3_teacher = list(map(id, teacher_model.decoder3.parameters()))
        base_params_teacher = filter(lambda p: id(p) not in params1_teacher + params2_teacher + params3_teacher, teacher_model.parameters())
        params_teacher = [{'params': base_params_teacher},
                  {'params': teacher_model.decoder1.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                  {'params': teacher_model.decoder2.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                  {'params': teacher_model.decoder3.parameters(), 'lr': lr / 100, 'weight_decay': wd}]
        optimizer_teacher = torch.optim.Adam(params_teacher, lr=lr, weight_decay=wd)

        params1 = list(map(id, student_model.decoder1.parameters()))
        params2 = list(map(id, student_model.decoder2.parameters()))
        params3 = list(map(id, student_model.decoder3.parameters()))
        base_params = filter(lambda p: id(p) not in params1 + params2 + params3, student_model.parameters())
        params = [{'params': base_params},
                  {'params': student_model.decoder1.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                  {'params': student_model.decoder2.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                  {'params': student_model.decoder3.parameters(), 'lr': lr / 100, 'weight_decay': wd}]
        optimizer_student = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    else:    
        optimizer_teacher = torch.optim.Adam(teacher_model.parameters(), lr=lr, weight_decay=wd)
        optimizer_student = torch.optim.Adam(student_model.parameters(), lr=lr, weight_decay=wd)
    
    print("{:*^50}".format("training start"))
    for epoch in range(num_epoch):

        if epoch <= 30:
            teacher_model.train()
            epoch_loss = 0
            step = 0
            loss = 0
            pos_loss = 0
            neg_loss = 0

            for index, batch in enumerate(dataloader_train):
                image, label, gt = batch
                image = image.to(device)
                label = label.to(device)
                gt = gt.to(device)
                loss1 = 0
                loss2 = 0
                loss3 = 0
                lossf = 0
                        
                side1, side2, side3, fusion, pixel_x = teacher_model(image)

                label = label.unsqueeze(2).unsqueeze(3)
                label = label.expand(len(label), 1, 256, 256)
                # loss_oeem = cross_entropy(fusion, label)

                for i in range(len(label)):
                    temp1 = label[i].sum().item()    
                    if temp1 != 0:
                        loss1 += dice(side1[i], label[i])
                        loss2 += dice(side2[i], label[i])
                        loss3 += dice(side3[i], label[i])
                        lossf += dice(fusion[i], label[i])
                    else:
                        loss1 += dice(-(side1[i]-1), -(label[i]-1))
                        loss2 += dice(-(side2[i]-1), -(label[i]-1))
                        loss3 += dice(-(side3[i]-1), -(label[i]-1))
                        lossf += dice(-(fusion[i]-1), -(label[i]-1))

                loss = loss1 + loss2 + loss3 + lossf

                optimizer_teacher.zero_grad()
                loss.backward()
                optimizer_teacher.step()

                epoch_loss += loss.item()
                step += 1
                    
            average_loss = epoch_loss / math.ceil(dataset_size // dataloader_train.batch_size)
            print("epoch %d loss:%0.4f" % (epoch, average_loss))

            if valid_fn is not None:
                teacher_model.eval()
                result = valid_fn(teacher_model, dataloader_valid, epoch, device)
                print('teacher: epoch %d loss:%.4f result:%.3f' % (epoch, average_loss, result))
                print('')
                if result > best_result:
                    best_result = result
                    torch.save(teacher_model.state_dict(), path_work + 'best_teacher_model.pth')
            torch.save(teacher_model.state_dict(), path_work + 'final_teacher_model.pth')
            
        else:
            if epoch == 31:
                    checkpoint = torch.load('./work/test/best_teacher_model.pth')
                    teacher_model.load_state_dict(checkpoint)
                    teacher_model.eval()
                    result= valid_fn(teacher_model, dataloader_valid, epoch, device)
                    print('teacher: epoch %d loss:%.4f result:%.3f' % (epoch, average_loss, result))
                    print('')
                    new_state_dict = teacher_model.state_dict()

            student_model.train()
            teacher_model.eval()

            epoch_loss = 0
            step = 0
            loss = 0
            pos_loss = 0
            neg_loss = 0
            for index, batch in enumerate(dataloader_train):
                image, label, gt = batch
                image = image.to(device)
                label = label.to(device)
                gt = gt.to(device)
                loss1 = 0
                loss2 = 0
                loss3 = 0
                lossf = 0
                side1, side2, side3, fusion, pixel_x = student_model(image)
                side1_t, side2_t, side3_t, fusion_t, pixel_x_t = teacher_model(image)
                loss_oeem = cross_entropy(fusion, fusion_t)


                for i in range(len(label)):
                    temp1 = label[i].sum().item()    
                    if temp1 != 0:
                        loss1 += dice(side1[i], fusion_t[i])
                        loss2 += dice(side2[i], fusion_t[i])
                        loss3 += dice(side3[i], fusion_t[i])
                        lossf += dice(fusion[i], fusion_t[i])
                    else:
                        loss1 += dice(-(side1[i]-1), -(gt[i]-1))
                        loss2 += dice(-(side2[i]-1), -(gt[i]-1))
                        loss3 += dice(-(side3[i]-1), -(gt[i]-1))
                        lossf += dice(-(fusion[i]-1), -(gt[i]-1))
                loss = loss1 + loss2 + loss3 + lossf + loss_oeem*dataloader_train.batch_size*2
                loss.backward()
                optimizer_student.step()
                #scheduler.step()
                optimizer_student.zero_grad()

                epoch_loss += loss.item()
                step += 1
            average_loss = epoch_loss / math.ceil(dataset_size // dataloader_train.batch_size)
            print("epoch %d loss:%0.4f" % (epoch, average_loss))
            print(pos_loss, neg_loss)


            if valid_fn is not None:
                student_model.eval()
                result = valid_fn(student_model, dataloader_valid, epoch, device)
                print('student: epoch %d loss:%.4f result:%.3f' % (epoch, average_loss, result))
                print('')
                if result > best_result:
                    best_result = result
                    torch.save(student_model.state_dict(), path_work + 'best_student_model.pth')

            torch.save(student_model.state_dict(), path_work + 'final_student_model.pth')


            if epoch>=60 and epoch % 30 == 0:
                teacher_model.eval()
                student_model.eval()
                state_dict_best_student = torch.load('./work/test/best_student_model.pth')
                state_dict_best_teacher = teacher_model.state_dict()#torch.load('./work/test/best_teacher_model.pth')
                
                teacher_model.load_state_dict(state_dict_best_student)
                teacher_model.eval()
                student_model.load_state_dict(state_dict_best_teacher)
                teacher_model.eval()
                
                print("parameter switch")
                result  = valid_fn(teacher_model, dataloader_valid, epoch, device)
                print('teacher: epoch %d loss:%.4f result:%.3f' % (epoch, average_loss, result))
                result = valid_fn(student_model, dataloader_valid, epoch, device)
                print('student: epoch %d loss:%.4f result:%.3f' % (epoch, average_loss, result))
                print('')


    print('best result: %.3f' % best_result)
