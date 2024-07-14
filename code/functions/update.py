import torch
import numpy as np
from sklearn import metrics
import cv2
from scipy.ndimage import gaussian_filter
from utils import gm

from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter

# from sklearn.metrics import precision_recall_curve

def dice(true, pred, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = (intersection+1) / (np.sum(pred) + np.sum(true)+1)
    return dice

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

def clean_filename(filename):
    filename =''.join([c for c in filename if c.isdigit()])

    return filename

def update(model, dataloader, test_num_pos, epoch, device, teacher = True):

    step = 0
    num = 0
    num_pos = 0
    num_neg = 0
    total_f = 0
    f1_pos_total = 0
    f1_neg_total = 0

    
    pr_list = []
    gt_list = []
    classfication_pred = []
    classfication_label = []
    classfication_pred_x4 = []
    weight = [0]

    with torch.no_grad():
        for image, label, list_img in dataloader:
            step += 1

            num += 1
    
            side1, side2, side3, fusion, pixel_x = model(image.to(device))                    
        

            pred = ((pixel_x[0].squeeze(0).squeeze(0).to('cpu').numpy() >= 0.5) + 0)

            
            label = label.squeeze(0).squeeze(0).to('cpu').numpy()




            pred_reshape = pred.reshape(-1)
            label_reshape = label.reshape(-1)
            temp1 = label_reshape.sum().item()



            if teacher == True:
                if temp1 != 0:
                    cv2.imwrite('./data/colon/train/gt/'+clean_filename(str(list_img[step-1]))+'.png',pixel_x[0][0][0].clone().to('cpu').numpy()*255)


                
            if temp1 != 0:
                f1_pos = metrics.f1_score(label_reshape, pred_reshape, pos_label=1)
                f1_pos_total += f1_pos 
                total_f += f1_pos 
                num_pos += 1
            else:
                f1_neg = metrics.f1_score(label_reshape, pred_reshape, pos_label=0)
                f1_neg_total += f1_neg
                total_f += f1_neg 
                num_neg += 1



        average_f = total_f/num

    print('pos f1: ', f1_pos_total/(num_pos+0.1), 'neg f1: ', f1_neg_total/(num_neg+0.1))

    return average_f