import random
import itertools
import numpy as np
import pandas as pd
import sklearn.metrics as metr
import torch
import matplotlib.pyplot as plt

from config import NORM_PARAM_DEPTH, NORM_PARAM_PATHS, MODEL_CONFIG
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, jaccard_score, hamming_loss, label_ranking_loss, coverage_error


norm_param_depth = NORM_PARAM_DEPTH["agia_napa"]
norm_param = np.load(NORM_PARAM_PATHS["agia_napa"])

crop_size = MODEL_CONFIG["crop_size"]
window_size = MODEL_CONFIG["window_size"]
stride = MODEL_CONFIG["stride"]

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w)
    x2 = x1 + w
    y1 = random.randint(0, H - h)
    y2 = y1 + h
    return x1, x2, y1, y2

def sliding_window(top, step=10, window_size=(20,20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
            
def count_sliding_window(top, step=10, window_size=(20,20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def calculate_metrics(predictions, gts):
    non_zero_mask = predictions != 0
    
    rmse = np.sqrt(np.mean(((predictions - gts) ** 2)[non_zero_mask]))
    mae = np.mean(np.abs((predictions - gts)[non_zero_mask]))
    std_dev = np.std((predictions - gts)[non_zero_mask])
    
    print("RMSE : {:.3f}m".format(rmse*-norm_param_depth))
    print("MAE : {:.3f}m".format(mae*-norm_param_depth))
    print("Std_Dev : {:.3f}m".format(std_dev*-norm_param_depth))
    print("---")
    
    return rmse,mae,std_dev

def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
        
        return tuple(results)
def metrics_marida(y_true,y_predicted):
    micro_prec = precision_score(y_true, y_predicted, average='micro')
    macro_prec = precision_score(y_true, y_predicted, average='macro')
    weight_prec = precision_score(y_true, y_predicted, average='weighted')
    
    micro_rec = recall_score(y_true, y_predicted, average='micro')
    macro_rec = recall_score(y_true, y_predicted, average='macro')
    weight_rec = recall_score(y_true, y_predicted, average='weighted')
        
    macro_f1 = f1_score(y_true, y_predicted, average="macro")
    micro_f1 = f1_score(y_true, y_predicted, average="micro")
    weight_f1 = f1_score(y_true, y_predicted, average="weighted")
        
    subset_acc = accuracy_score(y_true, y_predicted)
    
    iou_acc = jaccard_score(y_true, y_predicted, average='macro')

    info = {
            "macroPrec" : macro_prec,
            "microPrec" : micro_prec,
            "weightPrec" : weight_prec,
            "macroRec" : macro_rec,
            "microRec" : micro_rec,
            "weightRec" : weight_rec,
            "macroF1" : macro_f1,
            "microF1" : micro_f1,
            "weightF1" : weight_f1,
            "subsetAcc" : subset_acc,
            "IoU": iou_acc
            }
    
    return info
   
   
def confusion_matrix(y_gt, y_pred, labels):

    # compute metrics
    cm      = metr.confusion_matrix  (y_gt, y_pred)
    f1_macro= metr.f1_score          (y_gt, y_pred, average='macro')
    mPA      = metr.recall_score      (y_gt, y_pred, average='macro')
    OA      = metr.accuracy_score    (y_gt, y_pred)
    UA      = metr.precision_score   (y_gt, y_pred, average=None)
    PA      = metr.recall_score      (y_gt, y_pred, average=None)
    f1      = metr.f1_score          (y_gt, y_pred, average=None)
    IoC     = metr.jaccard_score     (y_gt, y_pred, average=None)
    mIoC     = metr.jaccard_score    (y_gt, y_pred, average='macro')
      
    # confusion matrix
    sz1, sz2 = cm.shape
    cm_with_stats             = np.zeros((sz1+4,sz2+2))
    cm_with_stats[0:-4, 0:-2] = cm
    cm_with_stats[-3  , 0:-2] = np.round(IoC,2)
    cm_with_stats[-2  , 0:-2] = np.round(UA,2)
    cm_with_stats[-1  , 0:-2] = np.round(f1,2)
    cm_with_stats[0:-4,   -1] = np.round(PA,2)
    
    cm_with_stats[-4  , 0:-2] = np.sum(cm, axis=0) 
    cm_with_stats[0:-4,   -2] = np.sum(cm, axis=1)
    
    # convert to list
    cm_list = cm_with_stats.tolist()
    
    # first row
    first_row = []
    first_row.extend (labels)
    first_row.append ('Sum')
    first_row.append ('Recall')
    
    # first col
    first_col = []
    first_col.extend(labels)
    first_col.append ('Sum')
    first_col.append ('IoU')
    first_col.append ('Precision')
    first_col.append ('F1-score')
    
    # fill rest of the text 
    idx = 0
    for sublist in cm_list:
        if   idx == sz1:
            sublist[-2]  = 'mPA:'
            sublist[-1]  = round(mPA,2)
            cm_list[idx] = sublist
        elif   idx == sz1+1:
            sublist[-2]  = 'mIoU:'
            sublist[-1]  = round(mIoC,2)
            cm_list[idx] = sublist
            
        elif idx == sz1+2:
            sublist[-2]  = 'OA:'
            sublist[-1]  = round(OA,2)
            cm_list[idx] = sublist
            
        elif idx == sz1+3:
            cm_list[idx] = sublist
            sublist[-2]  = 'F1-macro:'
            sublist[-1]  = round(f1_macro,2)    
        idx +=1
    
    # Convert to data frame
    df = pd.DataFrame(np.array(cm_list))
    df.columns = first_row
    df.index = first_col
    
    return df

def print_confusion_matrix_ML(confusion_matrix, class_label, ind_names, col_names):

    df_cm = pd.DataFrame(confusion_matrix, index=ind_names, columns=col_names)
    
    df_cm.index.name = class_label
    return df_cm
'''
def read_geotiff(filename, b):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(b)
    arr = band.ReadAsArray()
    return arr, ds

def write_geotiff(filename, arr, in_ds):
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)'''