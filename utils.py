import os
import cv2
import torch
import numpy as np
import random
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize01(x, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (x - x.min()) / (x.max() - x.min())
    else:
        return (x - min_value) / (max_value - min_value)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc

def evaluate_obj(results_obj):
    table, gt_px, pr_px, gt_sp, pr_sp = [], [], [], [], []
    auroc_sp, auroc_px, f1_sp, f1_px, aupro, ap_sp, ap_px = 0, 0, 0, 0, 0, 0, 0
    table.append(results_obj['cls_names'])
    
    if 'sp' in results_obj['metric']:
        gt_sp= np.array(results_obj['gt_sp'])
        pr_sp = np.array(results_obj['pr_sp'])
        
        auroc_sp = roc_auc_score(gt_sp, pr_sp)
        ap_sp = average_precision_score(gt_sp, pr_sp)
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
    
    if 'px' in results_obj['metric']:
        gt_px = np.array(results_obj['imgs_masks'])
        pr_px= np.array(results_obj['anomaly_maps'])
        
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        aupro = cal_pro_score(gt_px, pr_px)

    table.append(str(np.round(auroc_px * 100, decimals=2)))
    table.append(str(np.round(f1_px * 100, decimals=2)))
    table.append(str(np.round(ap_px * 100, decimals=2)))
    table.append(str(np.round(aupro * 100, decimals=2)))
    table.append(str(np.round(auroc_sp * 100, decimals=2)))
    table.append(str(np.round(f1_sp * 100, decimals=2)))
    table.append(str(np.round(ap_sp * 100, decimals=2)))
    
    return table, auroc_sp, auroc_px, f1_sp, f1_px, aupro, ap_sp, ap_px

def visualization(image_path_list, pr_px, category, vis_type, output_dir):
    def normalization01(img):
        return (img - img.min()) / (img.max() - img.min())
    print('visualization...')
    if vis_type == 'single_norm':
        # normalized per image
        for i, path in enumerate(image_path_list):
            anomaly_type = path.split('/')[-2]
            img_name = path.split('/')[-1]
            save_path = os.path.join(output_dir, 'vis', category, anomaly_type)
            
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, img_name)
            anomaly_map = pr_px[i].squeeze()
            anomaly_map = normalization01(anomaly_map)*255
            anomaly_map = cv2.applyColorMap(anomaly_map.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(save_path, anomaly_map)
    elif vis_type == 'all_norm':
        # normalized all image
        pr_px = normalization01(pr_px)
        for i, path in enumerate(image_path_list):
            anomaly_type = path.split('/')[-2]
            img_name = path.split('/')[-1]
            save_path = os.path.join(output_dir, 'vis', category, anomaly_type)
            vis = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            h, w, c = vis.shape
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, img_name)
            anomaly_map = pr_px[i].squeeze()*255
            vis = apply_ad_scoremap(vis, anomaly_map)
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, vis)
    else: # vis_type == 'no_norm':
        for i, path in enumerate(image_path_list):
            anomaly_type = path.split('/')[-2]
            img_name = path.split('/')[-1]
            save_path = os.path.join(output_dir, 'vis', category, anomaly_type)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, img_name)
            anomaly_map = pr_px[i].squeeze()
            anomaly_map = anomaly_map*255
            anomaly_map = cv2.applyColorMap(anomaly_map.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(save_path, anomaly_map)