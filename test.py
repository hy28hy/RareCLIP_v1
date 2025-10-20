import torch
import numpy as np
import os
import argparse
from torch.nn import functional as F
import torchvision.transforms as transforms
import logging
import time
from multiprocessing import Pool
from tqdm import tqdm
from tabulate import tabulate

from dataset import VisaDataset, MVTecDataset, BtadDataset, CombineDataset
from rareclip import RareCLIP
from rareclip_d import RareCLIP_d
from utils import setup_seed, evaluate_obj, visualization

import warnings
warnings.filterwarnings('ignore')

def test(args, default_args):
    # configs
    image_size = args.image_size
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    save_path = args.save_path
    test_set = args.test 
    if args.test == 'visa':
        save_path = save_path.replace('mvtec', 'visa')
    else:
        save_path = save_path.replace('mvtec', args.test)
    test_set_path = args.root_path + '../dataset/' + test_set
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')  # log
    
    if args.direct:
        model = RareCLIP_d(args)
    else:
        model = RareCLIP(args)
    preprocess = model.preprocess

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    instruction = 'python test.py'
    for arg, value in vars(args).items():
        if default_args[arg] != value and arg != 'save_path' and arg != 'gpu' and arg != 'wait':
            instruction += f' --{arg} {value}' if not isinstance(value, list) else f' --{arg} ' + ' '.join([str(i) for i in value])
    logger.info(instruction)

    # transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    
    if args.test == 'mvtec':
        TestDataset = MVTecDataset
    elif args.test == 'visa':
        TestDataset = VisaDataset
    elif args.test == 'btad':
        TestDataset = BtadDataset
        
    test_dataset = TestDataset()
    obj_list = test_dataset.CLSNAMES if args.obj is None else args.obj
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad(), torch.cuda.amp.autocast():
        results = {}
        inference_time = 0
        total_test_num = 0
        for obj in obj_list:
            results[obj] = {}
            results[obj]['metric'] = args.metric
            results[obj]['cls_names'] = obj
            results[obj]['imgs_masks'] = []
            results[obj]['anomaly_maps'] = []
            results[obj]['gt_sp'] = []
            results[obj]['pr_sp'] = []
            img_paths = []
            
            model.renew_memory()
            
            test_dataset = TestDataset(root=test_set_path, transform=preprocess, target_transform=transform, obj_name=obj, shuffle_seed=args.seed)

            if args.k_shot > 0:
                k_shot_dataset = TestDataset(root=test_set_path, transform=preprocess, target_transform=transform, set='train', k_shot=args.k_shot, obj_name=obj, shuffle_seed=args.seed)
                for image_info in torch.utils.data.DataLoader(k_shot_dataset, batch_size=1, shuffle=False):
                    input_image = image_info["img"].to(device)
                    model.process_image_and_update(input_image)

            total_test_num += len(test_dataset)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            desc = obj + f'({obj_list.index(obj) + 1}/{len(obj_list)})'
            
            for idx, image_info in enumerate(tqdm(test_dataloader, desc=desc, leave=False)):
                torch.cuda.empty_cache()
                gt_mask = image_info['img_mask']
                path = image_info["img_path"][0]
                img_paths.append(path)
                gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
                if args.resize_mask > 0:
                    gt_mask = F.interpolate(gt_mask, size=args.resize_mask, mode='nearest')
                results[obj]['imgs_masks'].append(gt_mask.squeeze().numpy())
                results[obj]['gt_sp'].append(image_info['anomaly'].item())
                
                input_image = image_info["img"].to(device)
                starter.record()
                anomaly_map, anomaly_score = model.process_image_and_update(input_image, update=args.online)
                ender.record()
                torch.cuda.synchronize()
                single_inference_time = starter.elapsed_time(ender)
                inference_time += single_inference_time

                if args.resize_mask > 0:
                    anomaly_map = F.interpolate(anomaly_map, size=args.resize_mask, mode='bilinear', align_corners=True)
                results[obj]['pr_sp'].append(anomaly_score.item())
                results[obj]['anomaly_maps'].append(anomaly_map.squeeze().cpu().numpy())
            if args.vis_type is not None:
                visualization(img_paths, np.array(results[obj]['anomaly_maps']), obj, args.vis_type, save_path)
    logger.info(f'inference time: {int(inference_time / total_test_num * 10) / 10}ms per image')
        
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_ls = []
    ap_sp_ls = []
    ap_px_ls = []

    pool_num = min(len(obj_list), args.pool_num)
    with Pool(pool_num) as pool:
        returns = list(tqdm(pool.imap_unordered(evaluate_obj, results.values()), desc='metric', total=len(obj_list), leave=False))
    
    for obj in obj_list:
        for table, auroc_sp, auroc_px, f1_sp, f1_px, aupro, ap_sp, ap_px in returns:
            if table[0] == obj:
                table_ls.append(table)
                auroc_sp_ls.append(auroc_sp)
                auroc_px_ls.append(auroc_px)
                f1_sp_ls.append(f1_sp)
                f1_px_ls.append(f1_px)
                aupro_ls.append(aupro)
                ap_sp_ls.append(ap_sp)
                ap_px_ls.append(ap_px)
                break

    # logger
    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=2)),
                    str(np.round(np.mean(f1_px_ls) * 100, decimals=2)), str(np.round(np.mean(ap_px_ls) * 100, decimals=2)),
                    str(np.round(np.mean(aupro_ls) * 100, decimals=2)), str(np.round(np.mean(auroc_sp_ls) * 100, decimals=2)),
                    str(np.round(np.mean(f1_sp_ls) * 100, decimals=2)), str(np.round(np.mean(ap_sp_ls) * 100, decimals=2))])
    results = tabulate(table_ls, headers=['objects', 'auroc_px', 'f1_px', 'ap_px', 'aupro', 'auroc_sp',
                                        'f1_sp', 'ap_sp'], tablefmt="pipe")
    logger.info("\n%s", results)

if __name__ == '__main__':
    root = './'
    exps_name = 'default'
    
    parser = argparse.ArgumentParser("RareCLIP", add_help=True)
    # path
    parser.add_argument("--root_path", type=str, default=root, help="root path")
    parser.add_argument("--save_path", type=str, default='exps/mvtec/'+exps_name, help='path to save results')
    parser.add_argument("--load_path", type=str, default='weights/visa_pretrained.pth', help='path to load TPB weight')
    # dataset
    parser.add_argument("--test", type=str, default='mvtec', help="test dataset name, mvtec or visa")
    parser.add_argument("--test_set_path", type=str, default='../dataset/mvtec', help="test dataset path")
    parser.add_argument("--seed", type=int, default=0, help="random seed to use")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers when test")
    parser.add_argument("--obj", type=str, nargs="+", default=None, help="class name")
    # model
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id to use")
    # hyper-parameter
    parser.add_argument("--features_list_text", type=int, nargs="+", default=[12, 16, 20, 24], help="features used for TPB")
    parser.add_argument("--features_list_rare", type=int, nargs="+", default=[6, 12, 18, 24], help="features used for PRB")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--keep_ftime", type=float, default=3, help="keep features time for RareCLIP, default 3*1369")
    parser.add_argument("--keep_fratio", type=float, default=0.333, help="keep features ratio for RareCLIP-d")
    parser.add_argument("--keep_snum", type=int, default=200, help="keep sim num")
    parser.add_argument("--keep_inum", type=int, default=1000, help="keep image feature num")
    parser.add_argument("--topk", type=int, default=3, help="knn")
    parser.add_argument("--LS_ratio", type=float, default=0.01, help="LS ratio")
    parser.add_argument("--Rs", type=int, default=1, help="Rs or not Rs")
    parser.add_argument("--Rs_freq", type=int, default=20, help="Rs freq")
    parser.add_argument("--max_Rs_num", type=int, default=8, help="max neighbors num in Rs")
    parser.add_argument("--Rs_temp", type=float, default=200, help="Rs temp")
    parser.add_argument("--text_temp", type=float, default=20, help="text temperature")
    parser.add_argument("--sigma", type=float, default=4, help="sigma for gaussian filter")
    parser.add_argument("--rare_thd", type=float, default=0.3, help="rarity_thd")
    parser.add_argument("--sampler", type=str, default='SCS', help="SCS, GCS, RS or KCS")
    # other
    parser.add_argument("--k_shot", type=int, default=0, help="k-normal-shot")
    parser.add_argument("--online", type=int, default=1, help="update or not when test, 1 for online, 0 for offline")
    parser.add_argument("--direct", type=int, default=0, help="use RareCLIP-d")
    parser.add_argument("--metric", type=str, default='px+sp', help="result metric, pixel- or sample- level")
    parser.add_argument("--resize_mask", type=int, default=256, help="resize pixel-level result to accelerate metric")
    parser.add_argument("--pool_num", type=int, default=3, help="number of process pools to accelerate metric")
    parser.add_argument("--vis_type", type=str, default=None, help="vis type:'no_norm', 'single_norm', 'all_norm'")
    parser.add_argument("--wait", type=int, default=0, help="minutes to wait")
    parser.add_argument("--other", type=str, default='', help="other thing")

    args = parser.parse_args()
    default_args = vars(parser.parse_args([]))
    if args.wait:
        time.sleep(60 * args.wait)
    setup_seed(args.seed)
    test(args, default_args)

