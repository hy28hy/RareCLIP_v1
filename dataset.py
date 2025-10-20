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
                self.data_all.extend(data_cls[:k_shot] if k_shot < len(data_cls) else data_cls)
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
                self.data_all.extend(data_cls[:k_shot] if k_shot < len(data_cls) else data_cls)
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
