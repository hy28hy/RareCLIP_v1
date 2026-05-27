import torch.utils.data as data
import json
import random
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torch
import os
import pandas as pd


class VisaDataset(data.Dataset):
    CLSNAMES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
                 "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
    
    def __init__(self, root=None, transform=None, target_transform=None, train_aug=0, set='test', k_shot=0, save_dir=None, obj_name=None, shuffle_seed=None):
        if root is None:
            return
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_aug = train_aug
        self.set = set

        self.data_all = []
        meta_path = f'{self.root}/meta.json'
        if not os.path.exists(meta_path):
            print('generate meta.json')
            self.make_meta_json()
        meta_info = json.load(open(meta_path, 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[set]

        self.cls_names = self.CLSNAMES if obj_name is None else [obj_name]
        for cls_name in self.cls_names:
            if k_shot > 0:
                data_cls = meta_info[cls_name]
                random.seed(shuffle_seed)
                random.shuffle(data_cls)
                self.data_all.extend(data_cls[:k_shot] if k_shot < len(data_cls) else data_cls)
            else:
                self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)


        if shuffle_seed is not None and k_shot <= 0:
            random.seed(shuffle_seed)
            random.shuffle(self.data_all)

    def make_meta_json(self):
        meta_path = f'{self.root}/meta.json'
        csv_data = pd.read_csv(f'{self.root}/split_csv/1cls.csv', header=0)
        columns = csv_data.columns  # [object, split, label, image, mask]
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_data = csv_data[csv_data[columns[0]] == cls_name]
            anno_data = pd.read_csv(f'{self.root}/{cls_name}/image_anno.csv', header=0)
            anno_columns = anno_data.columns # [image, label, mask]
            for phase in ['train', 'test']:
                cls_info = []
                cls_data_phase = cls_data[cls_data[columns[1]] == phase]
                cls_data_phase.index = list(range(len(cls_data_phase)))
                for idx in range(cls_data_phase.shape[0]):
                    data = cls_data_phase.loc[idx]
                    is_abnormal = True if data.iloc[2] == 'anomaly' else False
                    img_path = data.iloc[3]
                    specie = anno_data[anno_data[anno_columns[0]] == img_path].iloc[0,1]
                    info_img = dict(
                        img_path=img_path,
                        mask_path=data.iloc[4] if is_abnormal else '',
                        cls_name=cls_name,
                        specie_name=specie,
                        anomaly=1 if is_abnormal else 0,
                    )
                    cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        with open(meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

    def __len__(self):
        return self.length

    def get_cls_names(self):
        return self.cls_names

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
            img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
                
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            img_mask = self.target_transform(img_mask)

        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path)}


class MVTecDataset(data.Dataset):
    CLSNAMES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
                "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]

    def __init__(self, root=None, transform=None, target_transform=None, train_aug=0, set='test', k_shot=0, save_dir=None, obj_name=None, shuffle_seed=None):
        if root is None:
            return
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_aug = train_aug
        self.combine_rate = 0.2
        self.k_shot = k_shot
        self.set = set

        self.data_all = []
        meta_path = f'{self.root}/meta.json'
        if not os.path.exists(meta_path):
            print('generate meta.json')
            self.make_meta_json()
        meta_info = json.load(open(meta_path, 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[set]

        self.cls_names = self.CLSNAMES if obj_name is None else [obj_name]
        for cls_name in self.cls_names:
            if k_shot > 0:
                data_cls = meta_info[cls_name]
                random.seed(shuffle_seed)
                random.shuffle(data_cls)
                # 支持重复采样：如果样本不足 k_shot，则重复采样凑够
                if k_shot <= len(data_cls):
                    self.data_all.extend(data_cls[:k_shot])
                else:
                    # 重复采样：先取全部，再重复取直到凑够 k_shot
                    self.data_all.extend(data_cls)
                    remaining = k_shot - len(data_cls)
                    self.data_all.extend(data_cls[:remaining])
            else:
                if train_aug > 0 and cls_name in ['transistor', 'cable']:
                    cls_data = meta_info[cls_name]
                    for data in cls_data:
                        # 这两个异常可能导致模型区分前景和背景的能力下降
                        if data['specie_name'] not in ['misplaced', 'missing_cable']:
                            self.data_all.append(data)
                else:
                    self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        if shuffle_seed is not None and k_shot <= 0:
            random.seed(shuffle_seed)
            random.shuffle(self.data_all)
            
    def reverse(self):
        self.data_all.reverse()
        
    def shuffle_by_idxs(self, idxs):
        self.data_all = [self.data_all[i] for i in idxs]
        
    def select_by_idxs(self, idxs):
        self.data_all = [self.data_all[i] for i in idxs]
        self.length = len(self.data_all)

    def make_meta_json(self):
        meta_path = f'{self.root}/meta.json'
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                species.sort()
                for specie in species:
                    is_abnormal = specie != 'good'
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    mask_names = os.listdir(f'{cls_dir}/ground_truth/{specie}') if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        with open(meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

    def __len__(self):
        return self.length

    def get_cls_names(self):
        return self.cls_names

    def combine_img(self, cls_name):
        img_paths = os.path.join(self.root, cls_name, 'test')
        img_ls = []
        mask_ls = []
        for i in range(4):
            defect = os.listdir(img_paths)
            random_defect = random.choice(defect)
            while random_defect in ['misplaced', 'missing_cable']:
                random_defect = random.choice(defect)
            files = os.listdir(os.path.join(img_paths, random_defect))
            random_file = random.choice(files)
            img_path = os.path.join(img_paths, random_defect, random_file)
            mask_path = os.path.join(self.root, cls_name, 'ground_truth', random_defect, random_file[:3] + '_mask.png')
            img = Image.open(img_path)
            img_ls.append(img)
            if random_defect == 'good':
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(mask_path).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            mask_ls.append(img_mask)
        # image
        image_width, image_height = img_ls[0].size
        result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
        for i, img in enumerate(img_ls):
            row = i // 2
            col = i % 2
            x = col * image_width
            y = row * image_height
            result_image.paste(img, (x, y))

        # mask
        result_mask = Image.new("L", (2 * image_width, 2 * image_height))
        for i, img in enumerate(mask_ls):
            row = i // 2
            col = i % 2
            x = col * image_width
            y = row * image_height
            result_mask.paste(img, (x, y))

        return result_image, result_mask

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        if self.train_aug > 0 and random.random() < self.combine_rate:
            img, img_mask = self.combine_img(cls_name)
        else:
            img = Image.open(os.path.join(self.root, img_path))
            if anomaly == 0:
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        # transforms
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            img_mask = self.target_transform(img_mask)
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path)}
  
class BtadDataset(data.Dataset):
    CLSNAMES = ["01", "02", "03"]
    
    def __init__(self, root=None, transform=None, target_transform=None, train_aug=0, set='test', k_shot=0, save_dir=None, obj_name=None, shuffle_seed=None):
        if root is None:
            return
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_aug = train_aug
        
        self.data_all = []
        meta_path = f'{self.root}/meta.json'
        if not os.path.exists(meta_path):
            print('generate meta.json')
            self.make_meta_json()
        meta_info = json.load(open(meta_path, 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[set]

        self.cls_names = self.CLSNAMES if obj_name is None else [obj_name]
        for cls_name in self.cls_names:
            if k_shot > 0:
                data_cls = meta_info[cls_name]
                random.seed(shuffle_seed)
                random.shuffle(data_cls)
                # 支持重复采样：如果样本不足 k_shot，则重复采样凑够
                if k_shot <= len(data_cls):
                    self.data_all.extend(data_cls[:k_shot])
                else:
                    # 重复采样：先取全部，再重复取直到凑够 k_shot
                    self.data_all.extend(data_cls)
                    remaining = k_shot - len(data_cls)
                    self.data_all.extend(data_cls[:remaining])
            else:
                self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        if shuffle_seed is not None and k_shot <= 0:
            random.seed(shuffle_seed)
            random.shuffle(self.data_all)
            
    def reverse(self):
        self.data_all.reverse()
        
    def shuffle_by_idxs(self, idxs):
        self.data_all = [self.data_all[i] for i in idxs]
        
    def select_by_idxs(self, idxs):
        self.data_all = [self.data_all[i] for i in idxs]
        self.length = len(self.data_all)
        
    def make_meta_json(self):
        meta_path = f'{self.root}/meta.json'
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                species.sort()
                for specie in species:
                    is_abnormal = specie != 'ok'
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    mask_names = os.listdir(f'{cls_dir}/ground_truth/{specie}') if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        with open(meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

    def __len__(self):
        return self.length

    def get_cls_names(self):
        return self.cls_names

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
            img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
                
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            img_mask = self.target_transform(img_mask)

        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path)}
  
class CombineDataset(data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(set) for set in datasets]
        self.index2set = []
        self.index2set_index = []
        for i, length in enumerate(self.lengths):
            self.index2set += [i] * length
            self.index2set_index += range(length)
        self.length = sum(self.lengths)
  
    def __len__(self):
        return self.length
    
    def shuffle_by_idxs(self, idxs):
        self.index2set = [self.index2set[i] for i in idxs]
        self.index2set_index = [self.index2set_index[i] for i in idxs]

    def __getitem__(self, index):
        return self.datasets[self.index2set[index]][self.index2set_index[index]]
    
    def shuffle(self, start=0, seed=0):
        index_zip = list(zip(self.index2set, self.index2set_index))
        part = index_zip[start:]
        random.seed(seed)
        random.shuffle(part)
        index_zip[start:] = part
        self.index2set, self.index2set_index = zip(*index_zip)

# =================================================================
# 1. 统一医学异常数据集 (Brain, Liver, Retina)
# 适用结构: root/test/good/img, 或 root/test/good/000.png 均可兼容
# =================================================================
class MedicalAnomalyDataset(data.Dataset):
    def __init__(self, root=None, transform=None, target_transform=None, train_aug=0, set='test', k_shot=0, save_dir=None, obj_name=None, shuffle_seed=None):
        if root is None:
            return
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_aug = train_aug
        self.set = set
        
        # 自动获取根目录的名称作为类别名 (例如 "Brain", "Liver")
        dataset_name = os.path.basename(os.path.normpath(self.root))
        self.CLSNAMES = [dataset_name]

        self.data_all = []
        meta_path = f'{self.root}/meta.json'
        if not os.path.exists(meta_path):
            print(f'Generating meta.json for {dataset_name}...')
            self.make_meta_json()
            
        meta_info = json.load(open(meta_path, 'r'))
        if set in meta_info:
            meta_info = meta_info[set]
        else:
            meta_info = {dataset_name: []}

        self.cls_names = self.CLSNAMES if obj_name is None else [obj_name]
        for cls_name in self.cls_names:
            if k_shot > 0:
                data_cls = meta_info.get(cls_name, [])
                random.seed(shuffle_seed)
                random.shuffle(data_cls)
                self.data_all.extend(data_cls[:k_shot] if k_shot < len(data_cls) else data_cls)
            else:
                self.data_all.extend(meta_info.get(cls_name, []))
        self.length = len(self.data_all)

        if shuffle_seed is not None and k_shot <= 0:
            random.seed(shuffle_seed)
            random.shuffle(self.data_all)

    def make_meta_json(self):
        meta_path = f'{self.root}/meta.json'
        info = dict(train={}, test={}, valid={})
        dataset_name = self.CLSNAMES[0]
        
        for phase in ['train', 'test', 'valid']:
            cls_info = []
            phase_dir = f'{self.root}/{phase}'
            if not os.path.exists(phase_dir):
                continue
                
            # 获取下面所有的目录 (过滤掉可能存在的非目录文件)
            species = [d for d in os.listdir(phase_dir) if os.path.isdir(os.path.join(phase_dir, d))]
            species.sort()
            
            for specie in species:
                is_abnormal = (specie != 'good') 
                specie_dir = f'{phase_dir}/{specie}'
                
                # 智能判断：图片是在 img/ 子文件夹里，还是直接在当前文件夹里？
                if os.path.exists(f'{specie_dir}/img'):
                    img_dir = f'{specie_dir}/img'
                    img_rel_prefix = f'{phase}/{specie}/img'
                else:
                    img_dir = specie_dir
                    img_rel_prefix = f'{phase}/{specie}'

                # 过滤出真正的图片文件（防止误读到 label 文件夹）
                valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
                img_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and f.lower().endswith(valid_exts)]
                img_names.sort()

                for img_name in img_names:
                    # 确定掩码的相对路径 (假设它在 label/ 目录下，如果不存在会在后续生成全黑Mask)
                    mask_rel_path = f'{phase}/{specie}/label/{img_name}' if is_abnormal else ''
                    
                    info_img = dict(
                        img_path=f'{img_rel_prefix}/{img_name}',
                        mask_path=mask_rel_path,
                        cls_name=dataset_name,
                        specie_name=specie,
                        anomaly=1 if is_abnormal else 0,
                    )
                    cls_info.append(info_img)
            info[phase][dataset_name] = cls_info
            
        with open(meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

    def __len__(self): return self.length
    def get_cls_names(self): return self.cls_names

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly']
        
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            mask_full_path = os.path.join(self.root, mask_path)
            if os.path.exists(mask_full_path):
                img_mask = np.array(Image.open(mask_full_path).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            else:
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
                
        if self.transform is not None: img = self.transform(img)
        if self.target_transform is not None: img_mask = self.target_transform(img_mask)

        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': os.path.join(self.root, img_path)}

# =================================================================
# 2. 息肉分割数据集 (ColonDB, ClinicDB, Kvasir, CVC-300)
# 适用结构: root/images, root/masks
# =================================================================
class PolypDataset(data.Dataset):
    def __init__(self, root=None, transform=None, target_transform=None, train_aug=0, set='test', k_shot=0, save_dir=None, obj_name=None, shuffle_seed=None):
        if root is None:
            return
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        dataset_name = os.path.basename(os.path.normpath(self.root))
        self.CLSNAMES = [dataset_name]

        self.data_all = []
        meta_path = f'{self.root}/meta.json'
        if not os.path.exists(meta_path):
            self.make_meta_json()
            
        meta_info = json.load(open(meta_path, 'r'))
        # 息肉通常作为 zero-shot 测试集，全部归为 test
        meta_info = meta_info.get('test', {dataset_name: []})

        self.cls_names = self.CLSNAMES if obj_name is None else [obj_name]
        for cls_name in self.cls_names:
            if k_shot > 0:
                data_cls = meta_info.get(cls_name, [])
                random.seed(shuffle_seed)
                random.shuffle(data_cls)
                # 支持重复采样：如果样本不足 k_shot，则重复采样凑够
                if k_shot <= len(data_cls):
                    self.data_all.extend(data_cls[:k_shot])
                else:
                    # 重复采样：先取全部，再重复取直到凑够 k_shot
                    self.data_all.extend(data_cls)
                    remaining = k_shot - len(data_cls)
                    self.data_all.extend(data_cls[:remaining])
            else:
                self.data_all.extend(meta_info.get(cls_name, []))
        self.length = len(self.data_all)

    def make_meta_json(self):
        meta_path = f'{self.root}/meta.json'
        info = dict(train={}, test={})
        dataset_name = self.CLSNAMES[0]
        
        img_dir = os.path.join(self.root, 'images')
        mask_dir = os.path.join(self.root, 'masks')
        
        cls_info = []
        if os.path.exists(img_dir):
            img_names = os.listdir(img_dir)
            img_names.sort()
            for img_name in img_names:
                # 兼容同名掩码或 _mask 后缀掩码
                name_base, ext = os.path.splitext(img_name)
                if os.path.exists(os.path.join(mask_dir, img_name)):
                    mask_rel_path = f'masks/{img_name}'
                elif os.path.exists(os.path.join(mask_dir, name_base + '_mask' + ext)):
                    mask_rel_path = f'masks/{name_base}_mask{ext}'
                else:
                    mask_rel_path = ''
                    
                cls_info.append(dict(
                    img_path=f'images/{img_name}',
                    mask_path=mask_rel_path,
                    cls_name=dataset_name,
                    specie_name='polyp',
                    anomaly=1 # 此类数据集默认图像均含异常区域
                ))
        
        info['test'][dataset_name] = cls_info
        info['train'][dataset_name] = []
        with open(meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

    def __len__(self): return self.length
    def get_cls_names(self): return self.cls_names

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly']
        
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        
        mask_full_path = os.path.join(self.root, mask_path)
        if mask_path and os.path.exists(mask_full_path):
            img_mask = np.array(Image.open(mask_full_path).convert('L')) > 0
            img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        else:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
                
        if self.transform is not None: img = self.transform(img)
        if self.target_transform is not None: img_mask = self.target_transform(img_mask)

        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': os.path.join(self.root, img_path)}


# =================================================================
# 3. DAGM 数据集
# =================================================================
class DAGMDataset(data.Dataset):
    CLSNAMES = [f"Class{i}" for i in range(1, 11)]

    def __init__(self, root=None, transform=None, target_transform=None, train_aug=0, set='test', k_shot=0, save_dir=None, obj_name=None, shuffle_seed=None):
        if root is None:
            return
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_aug = train_aug
        self.set = set

        self.data_all = []
        meta_path = f'{self.root}/meta.json'
        if not os.path.exists(meta_path):
            self.make_meta_json()
            
        meta_info = json.load(open(meta_path, 'r'))
        meta_info = meta_info.get(set, {cls: [] for cls in self.CLSNAMES})

        self.cls_names = self.CLSNAMES if obj_name is None else [obj_name]
        for cls_name in self.cls_names:
            if k_shot > 0:
                data_cls = meta_info.get(cls_name, [])
                random.seed(shuffle_seed)
                random.shuffle(data_cls)
                if k_shot <= len(data_cls):
                    self.data_all.extend(data_cls[:k_shot])
                else:
                    self.data_all.extend(data_cls)
                    remaining = k_shot - len(data_cls)
                    self.data_all.extend(data_cls[:remaining])
            else:
                self.data_all.extend(meta_info.get(cls_name, []))
        self.length = len(self.data_all)

    def make_meta_json(self):
        meta_path = f'{self.root}/meta.json'
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                # Check both lowercase and capitalized phase names
                phase_dir = f'{cls_dir}/{phase}'
                actual_phase = phase  # default to lowercase
                if not os.path.exists(phase_dir):
                    phase_dir = f'{cls_dir}/{phase.capitalize()}'
                    actual_phase = phase.capitalize()  # use capitalized version
                if not os.path.exists(phase_dir): continue
                
                # List all files in phase_dir (no species subdirectories)
                img_names = os.listdir(phase_dir)
                img_names.sort()
                
                for img_name in img_names:
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                        continue
                    
                    # Check if there's a corresponding mask in ground_truth folder
                    # Try different possible mask paths
                    mask_path = ''
                    possible_mask_paths = [
                        f'{cls_name}/ground_truth/{actual_phase}/{img_name}',
                        f'{cls_name}/ground_truth/{img_name}',
                    ]
                    
                    for possible_mask in possible_mask_paths:
                        if os.path.exists(os.path.join(self.root, possible_mask)):
                            mask_path = possible_mask
                            break
                    
                    # If mask exists, it's abnormal (anomaly=1), otherwise normal (anomaly=0)
                    is_abnormal = mask_path != ''
                    
                    info_img = dict(
                        img_path=f'{cls_name}/{actual_phase}/{img_name}',
                        mask_path=mask_path,
                        cls_name=cls_name,
                        specie_name='',
                        anomaly=1 if is_abnormal else 0,
                    )
                    cls_info.append(info_img)
                
                info[phase][cls_name] = cls_info
        with open(meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

    def __len__(self): return self.length
    def get_cls_names(self): return self.cls_names

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly']
        
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            mask_full_path = os.path.join(self.root, mask_path)
            if mask_path and os.path.exists(mask_full_path):
                img_mask = np.array(Image.open(mask_full_path).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            else:
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
                
        if self.transform is not None: img = self.transform(img)
        if self.target_transform is not None: img_mask = self.target_transform(img_mask)
            
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': os.path.join(self.root, img_path)}


# =================================================================
# 4. DTD-Synthetic 数据集
# =================================================================
class DTDSyntheticDataset(data.Dataset):
    CLSNAMES = ["Blotchy_099", "Fibrous_183", "Marbled_078", "Matted_069", "Mesh_114", 
                "Perforated_037", "Stratified_154", "Woven_001", "Woven_068", "Woven_104", "Woven_125"]

    def __init__(self, root=None, transform=None, target_transform=None, train_aug=0, set='test', k_shot=0, save_dir=None, obj_name=None, shuffle_seed=None):
        if root is None:
            return
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_aug = train_aug
        self.set = set

        self.data_all = []
        meta_path = f'{self.root}/meta.json'
        if not os.path.exists(meta_path):
            self.make_meta_json()
            
        meta_info = json.load(open(meta_path, 'r'))
        meta_info = meta_info.get(set, {cls: [] for cls in self.CLSNAMES})

        self.cls_names = self.CLSNAMES if obj_name is None else [obj_name]
        for cls_name in self.cls_names:
            if k_shot > 0:
                data_cls = meta_info.get(cls_name, [])
                random.seed(shuffle_seed)
                random.shuffle(data_cls)
                if k_shot <= len(data_cls):
                    self.data_all.extend(data_cls[:k_shot])
                else:
                    self.data_all.extend(data_cls)
                    remaining = k_shot - len(data_cls)
                    self.data_all.extend(data_cls[:remaining])
            else:
                self.data_all.extend(meta_info.get(cls_name, []))
        self.length = len(self.data_all)

    def make_meta_json(self):
        meta_path = f'{self.root}/meta.json'
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                # Check both lowercase and capitalized phase names
                phase_dir = f'{cls_dir}/{phase}'
                if not os.path.exists(phase_dir):
                    phase_dir = f'{cls_dir}/{phase.capitalize()}'
                if not os.path.exists(phase_dir): continue
                
                # DTD-Synthetic structure: phase/good and phase/bad directories
                species = os.listdir(phase_dir)
                species.sort()
                for specie in species:  # specie should be 'good' or 'bad'
                    specie_dir = f'{phase_dir}/{specie}'
                    if not os.path.isdir(specie_dir): continue
                    
                    is_abnormal = (specie == 'bad')  # bad = abnormal, good = normal
                    img_names = os.listdir(specie_dir)
                    img_names.sort()
                    
                    for img_name in img_names:
                        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                            continue
                        
                        # Determine mask path for abnormal samples
                        mask_path = ''
                        if is_abnormal:
                            # Try to find mask in ground_truth/bad directory
                            possible_mask = f'{cls_name}/ground_truth/bad/{img_name}'
                            if os.path.exists(os.path.join(self.root, possible_mask)):
                                mask_path = possible_mask
                        
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=mask_path,
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                
                info[phase][cls_name] = cls_info
        with open(meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

    def __len__(self): return self.length
    def get_cls_names(self): return self.cls_names

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly']
        
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            mask_full_path = os.path.join(self.root, mask_path)
            if mask_path and os.path.exists(mask_full_path):
                img_mask = np.array(Image.open(mask_full_path).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            else:
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
                
        if self.transform is not None: img = self.transform(img)
        if self.target_transform is not None: img_mask = self.target_transform(img_mask)
            
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': os.path.join(self.root, img_path)}


# =================================================================
# 5. SDD 数据集 (Steel Defect Detection)
# =================================================================
class SDDDataset(data.Dataset):
    CLSNAMES = [f"kos{i:02d}" for i in range(1, 18)]

    def __init__(self, root=None, transform=None, target_transform=None, train_aug=0, set='test', k_shot=0, save_dir=None, obj_name=None, shuffle_seed=None):
        if root is None:
            return
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_aug = train_aug
        self.set = set

        self.data_all = []
        meta_path = f'{self.root}/meta.json'
        if not os.path.exists(meta_path):
            self.make_meta_json()
            
        meta_info = json.load(open(meta_path, 'r'))
        meta_info = meta_info.get(set, {cls: [] for cls in self.CLSNAMES})

        self.cls_names = self.CLSNAMES if obj_name is None else [obj_name]
        for cls_name in self.cls_names:
            if k_shot > 0:
                data_cls = meta_info.get(cls_name, [])
                random.seed(shuffle_seed)
                random.shuffle(data_cls)
                if k_shot <= len(data_cls):
                    self.data_all.extend(data_cls[:k_shot])
                else:
                    self.data_all.extend(data_cls)
                    remaining = k_shot - len(data_cls)
                    self.data_all.extend(data_cls[:remaining])
            else:
                self.data_all.extend(meta_info.get(cls_name, []))
        self.length = len(self.data_all)

    def make_meta_json(self):
        meta_path = f'{self.root}/meta.json'
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            cls_info = []
            if not os.path.exists(cls_dir): continue
            img_names = os.listdir(cls_dir)
            img_names.sort()
            for img_name in img_names:
                # Only process image files, skip label files (containing '_label.')
                if (img_name.lower().endswith(('.jpg', '.jpeg', '.png')) and 
                    '_label.' not in img_name):
                    # SDD 标签文件名相同但扩展名是 .bmp
                    name_base = os.path.splitext(img_name)[0]
                    mask_name = f'{name_base}_label.bmp'
                    mask_path = f'{cls_name}/{mask_name}'
                    
                    # Check if mask file exists
                    mask_full_path = os.path.join(self.root, mask_path)
                    if not os.path.exists(mask_full_path):
                        mask_path = ''  # No mask found
                    
                    info_img = dict(
                        img_path=f'{cls_name}/{img_name}',
                        mask_path=mask_path,
                        cls_name=cls_name,
                        specie_name='good',
                        anomaly=1 if mask_path else 0,  # anomaly=1 if mask exists
                    )
                    cls_info.append(info_img)
            info['train'][cls_name] = cls_info  # Put in train for training
            info['test'][cls_name] = cls_info  # Also put in test for testing
        with open(meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

    def __len__(self): return self.length
    def get_cls_names(self): return self.cls_names

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly']
        
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        
        mask_full_path = os.path.join(self.root, mask_path)
        if os.path.exists(mask_full_path):
            img_mask = np.array(Image.open(mask_full_path).convert('L')) > 0
            img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        else:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
                
        if self.transform is not None: img = self.transform(img)
        if self.target_transform is not None: img_mask = self.target_transform(img_mask)
            
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': os.path.join(self.root, img_path)}


# =================================================================
# 6. MPDD 数据集
# =================================================================
class MpddDataset(data.Dataset):
    CLSNAMES = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']

    def __init__(self, root=None, transform=None, target_transform=None, train_aug=0, set='test', k_shot=0, save_dir=None, obj_name=None, shuffle_seed=None):
        if root is None:
            return
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train_aug = train_aug
        self.set = set

        self.data_all = []
        meta_path = f'{self.root}/meta.json'
        if not os.path.exists(meta_path):
            self.make_meta_json()
            
        meta_info = json.load(open(meta_path, 'r'))
        meta_info = meta_info.get(set, {cls: [] for cls in self.CLSNAMES})

        self.cls_names = self.CLSNAMES if obj_name is None else [obj_name]
        for cls_name in self.cls_names:
            if k_shot > 0:
                data_cls = meta_info.get(cls_name, [])
                random.seed(shuffle_seed)
                random.shuffle(data_cls)
                # 支持重复采样：如果样本不足 k_shot，则重复采样凑够
                if k_shot <= len(data_cls):
                    self.data_all.extend(data_cls[:k_shot])
                else:
                    # 重复采样：先取全部，再重复取直到凑够 k_shot
                    self.data_all.extend(data_cls)
                    remaining = k_shot - len(data_cls)
                    self.data_all.extend(data_cls[:remaining])
            else:
                self.data_all.extend(meta_info.get(cls_name, []))
        self.length = len(self.data_all)

    def make_meta_json(self):
        meta_path = f'{self.root}/meta.json'
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                if not os.path.exists(f'{cls_dir}/{phase}'): continue
                species = os.listdir(f'{cls_dir}/{phase}')
                species.sort()
                for specie in species:
                    is_abnormal = specie != 'good'
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    mask_names = os.listdir(f'{cls_dir}/ground_truth/{specie}') if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}' if is_abnormal and mask_names else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        with open(meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

    def __len__(self): return self.length
    def get_cls_names(self): return self.cls_names

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly']
        
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            mask_full_path = os.path.join(self.root, mask_path)
            if mask_path and os.path.exists(mask_full_path):
                img_mask = np.array(Image.open(mask_full_path).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            else:
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
                
        if self.transform is not None: img = self.transform(img)
        if self.target_transform is not None: img_mask = self.target_transform(img_mask)
            
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': os.path.join(self.root, img_path)}