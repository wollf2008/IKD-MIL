import torch
import numpy as np
from sklearn import metrics
import cv2
from scipy.ndimage import gaussian_filter
import torch.nn as nn
import math
import torchvision
import torchvision.transforms as transforms
import PIL
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from matplotlib import cm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
    
def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

def calculate_iou(target, prediction, binary_label):
    target_mask = np.equal(target, binary_label)
    prediction_mask = np.equal(prediction, binary_label)
    
    intersection = np.logical_and(target_mask, prediction_mask)
    union = np.logical_or(target_mask, prediction_mask)
    union_sum = np.sum(union)
    if union_sum == 0:
        return 1.0 if np.sum(intersection) == 0 else 0.0
    
    iou_score = np.sum(intersection) / union_sum
    return iou_score

def hausdorff(a, b):
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    contours_a, hierarchy_a = cv2.findContours(a.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, hierarchy_b = cv2.findContours(b.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    temp1 = None
    for i in range(len(contours_a)):
        if i == 0:
            temp1 = contours_a[0]
        else:
            temp1 = np.concatenate((temp1, contours_a[i]), axis=0)
    if temp1 is not None:
        contours_a = temp1
    else:
        contours_a = np.zeros((1, 1, 2), dtype=int)

    temp2 = None
    for i in range(len(contours_b)):
        if i == 0:
            temp2 = contours_b[0]
        else:
            temp2 = np.concatenate((temp2, contours_b[i]), axis=0)
    if temp2 is not None:
        contours_b = temp2
    else:
        contours_b = np.zeros((1, 1, 2), dtype=int)

    hausdorff_distance = hausdorff_sd.computeDistance(contours_a, contours_b)
    return hausdorff_distance

def valid(model, dataloader, epoch, device):
    
    step = 0
    num = 0
    num_pos = 0
    num_neg = 0
    total_f = 0
    total_hd = 0
    total_iou = 0
    iou_pos = 0
    iou_neg = 0
    f1_pos_total = 0
    f1_neg_total = 0


    with torch.no_grad():
        for image, label in dataloader:
            step += 1
            num += 1
    
            side1, side2, side3, fusion, pixel_x = model(image.to(device))
            pixel_x0 = pixel_x[0].squeeze(0).squeeze(0).to('cpu').numpy()
            pixel_x0 = gaussian_filter(pixel_x0, sigma=3)
            pred  = (pixel_x0 >= 0.5) + 0
            label = label.squeeze(0).squeeze(0).to('cpu').numpy()
            

            # if step%1 == 0 and step <= 10:
            #     img = cv2.cvtColor(image.permute(0, 2, 3, 1).cpu().numpy()[0], cv2.COLOR_BGR2RGB)
            #     img = np.uint8(min_max_norm(img) * 255)
            #
            #     cv2.imwrite('./valid_results/'+str(epoch)+'_img_'+str(step)+'.png',img)
            #     cv2.imwrite('./valid_results/'+str(epoch)+'_gt_'+str(step)+'.png',label*255)
            #     cv2.imwrite('./valid_results/'+str(epoch)+'_predx_'+str(step)+'.png',pixel_x[0][0][0].clone().to('cpu').numpy()*255)
            #     cv2.imwrite('./valid_results/'+str(epoch)+'_predx1_'+str(step)+'.png',pixel_x[1][0][0].clone().to('cpu').numpy()*255)
            #     cv2.imwrite('./valid_results/'+str(epoch)+'_predx2_'+str(step)+'.png',pixel_x[2][0][0].clone().to('cpu').numpy()*255)
            #     cv2.imwrite('./valid_results/'+str(epoch)+'_predx3_'+str(step)+'.png',pixel_x[3][0][0].clone().to('cpu').numpy()*255)
            #     cv2.imwrite('./valid_results/'+str(epoch)+'_pred_binary_'+str(step)+'.png',pred*255)


            pred_reshape = pred.reshape(-1)
            label_reshape = label.reshape(-1)
            temp1 = label_reshape.sum().item()
                
            if temp1 != 0:
                f1_pos = metrics.f1_score(label_reshape, pred_reshape, pos_label=1)
                f1_pos_total += f1_pos 
                total_f += f1_pos 
                num_pos += 1


                hausdorff_distance = hausdorff(pred, label)
                total_hd += hausdorff_distance

                iou_score_pos = calculate_iou(label, pred, binary_label = 1)
                iou_pos += iou_score_pos
                total_iou += iou_score_pos


            else:
                f1_neg = metrics.f1_score(label_reshape, pred_reshape, pos_label=0)
                f1_neg_total += f1_neg
                total_f += f1_neg 
                num_neg += 1

                iou_score_neg = calculate_iou(label, pred, binary_label = 0)
                iou_neg += iou_score_neg
                total_iou += iou_score_neg


        average_f = total_f/num
        average_hd = total_hd / num*2
        average_iou = total_iou/num


    print('pos f1: ', f1_pos_total/(num_pos+1), 'neg f1: ', f1_neg_total/(num_neg+1))
    print('pos iou: ', iou_pos/(num_pos+1), 'neg iou: ', iou_neg/(num_neg+1), 'total iou: ', average_iou)
    print('hd: ', average_hd)

    return average_f