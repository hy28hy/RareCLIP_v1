import os
import torch
import torch.nn as nn
import numpy as np
import random
import json
import argparse
from torch.nn import functional as F
import torchvision.transforms as transforms
import logging
from tqdm import tqdm

import open_clip
from dataset import VisaDataset, MVTecDataset
from loss import FocalLoss, BinaryDiceLoss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def encode_text_with_prompt_ensemble(model, tokenizer, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format('object') for state in prompt_state[i]] + [state.format('texture') for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenizer(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device)

    return text_features

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            tokens[i] = self.fc[i](tokens[i])
        return tokens


def train(args, default_args):
    # configs
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')  # log

    # model configs
    features_list = args.features_list
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)

    # clip model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, image_size, pretrained=args.pretrained, cache_dir='../cache')
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record instruction
    instruction = 'python train.py'
    for arg, value in vars(args).items():
        if default_args[arg] != value and arg != 'save_path' and arg != 'gpu':
            instruction += f' --{arg} {value}'
    logger.info(instruction)

    # transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    # datasets
    train = args.train
    train_set_path = args.train_set_path
    if train == 'mvtec':
        train_data = MVTecDataset(root=train_set_path, transform=preprocess, target_transform=transform,
                                  train_aug=args.aug_rate, set='test')
    else:
        train_data = VisaDataset(root=train_set_path, transform=preprocess, target_transform=transform,
                                  set='test')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    

    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(model, tokenizer, device)
            

    # linear layer
    trainable_layer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],len(args.features_list)).to(device)
    optimizer = torch.optim.Adam(list(trainable_layer.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_mse = nn.MSELoss()
      

    for epoch in range(epochs):
        loss_list = []
        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False) as tepoch:
            for items in tepoch:
                image = items['img'].to(device)
                label = items['anomaly'].to(device,dtype=torch.float32)
                cls_name = items['cls_name']
                B = len(cls_name)
                gt = items['img_mask'].squeeze().to(device)
                gt[gt > 0.5], gt[gt <= 0.5] = 1, 0
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        image_features, patch_tokens = model.encode_image(image, features_list)
                        patch_tokens = [p[:,1:,:] for p in patch_tokens]
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        
                    patch_tokens = trainable_layer(patch_tokens)
                        
                    if args.feature_align:
                        all_patch = torch.cat(patch_tokens, dim=1)
                        all_patch_norm = all_patch / all_patch.norm(dim=-1, keepdim=True)
                        sim = (all_patch_norm @ image_features.unsqueeze(dim=-1)).flatten()
                    
                    anomaly_maps = []
                    for layer in range(len(patch_tokens)):                  
                        patch_token = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
                        B, L, C = patch_tokens[layer].shape
                        H = int(np.sqrt(L))
                        anomaly_map = (20.0 * patch_token @ text_features)
                        anomaly_map = anomaly_map.permute(0, 2, 1).contiguous().view(B, 2, H, H)
                        anomaly_map = F.interpolate(anomaly_map, size=image_size, mode='bilinear', align_corners=True)
                        anomaly_maps.append(anomaly_map.softmax(dim=1))

                # losses
                loss = 0
                if args.feature_align:
                    loss += loss_mse(sim, torch.ones_like(sim))
                for num in range(len(anomaly_maps)):
                    loss += loss_focal(anomaly_maps[num], gt)
                    loss += loss_dice(anomaly_maps[num][:, 1, :, :], gt)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                tepoch.set_postfix(loss=loss.item())


        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        # save model
        if (args.save_freq == -1 and epoch + 1 == epochs) or (args.save_freq > 0 and (epoch + 1) % args.save_freq == 0):
            ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({'trainable_linearlayer': trainable_layer.state_dict()}, ckp_path)

if __name__ == '__main__':
    root = './'
    exps_name = 'train'
    
    parser = argparse.ArgumentParser("RareCLIP", add_help=True)
    # path
    parser.add_argument("--train_set_path", type=str, default=root+'../dataset/mvtec', help="train dataset path")
    parser.add_argument("--save_path", type=str, default=root+'exps/mvtec'+exps_name, help='path to save results')
    parser.add_argument("--config_path", type=str, default=root+'open_clip/model_configs/ViT-L-14-336.json', help="model configs")
    # model
    parser.add_argument("--train", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[12, 16, 20, 24], help="features used")
    # hyper-parameter
    parser.add_argument("--gpu", type=int, default=6, help="gpu id to use")
    parser.add_argument("--epoch", type=int, default=5, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--aug_rate", type=float, default=0.2, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=-1, help="save frequency, 0 for no save, -1 for last save")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--feature_align", type=int, default=1, help="0 or 1")
    parser.add_argument("--other", type=str, default='', help="other thing")
    args = parser.parse_args()
    default_args = vars(parser.parse_args([]))

    setup_seed(args.seed)
    train(args, default_args)

