import torch
import torch.nn as nn
import numpy as np
import json
from torch.nn import functional as F
import open_clip
from sampler import GreedyCoresetSampler
import torchvision.transforms as transforms

def normalize01(x, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (x - x.min()) / (x.max() - x.min())
    else:
        return (x - min_value) / (max_value - min_value)

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            tokens[i] = self.fc[i](tokens[i])
        return tokens
    
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

class RareCLIP_d():
    def __init__(self, args) -> None:
        self.args = args
        self.image_size = args.image_size
        self.device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
        model_name = args.model
        with open('open_clip/model_configs/' + model_name + '.json', 'r') as f:
            model_configs = json.load(f)
        
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, self.image_size, pretrained=self.args.pretrained, cache_dir='../cache')
        self.clip_model.to(self.device)
        
        tokenizer = open_clip.get_tokenizer(model_name)
        with torch.cuda.amp.autocast(), torch.no_grad():
            self.linear_layer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'], len(args.features_list_text)).to(self.device)
            checkpoint = torch.load(args.load_path, map_location=torch.device('cpu') if self.device == 'cpu' else lambda storage, loc: storage.cuda(self.device))
            self.linear_layer.load_state_dict(checkpoint["trainable_linearlayer"])
            
            self.text_features = encode_text_with_prompt_ensemble(self.clip_model, tokenizer, self.device)
            torch.cuda.empty_cache()

        self.foreground_map = None
        self.text_temp = args.text_temp
        self.rare_map_mean = 0
        self.text_map_mean = 0
        self.foreground_ratio = 0.5
        
        self.uni_features_list = sorted(list(set(args.features_list_text).union(args.features_list_rare)))
        self.rare_idx = [self.uni_features_list.index(l) for l in args.features_list_rare]
        self.text_idx = [self.uni_features_list.index(l) for l in args.features_list_text]
        self.l_list = range(len(self.rare_idx))
        self.H, self.W = self.image_size // model_configs['vision_cfg']['patch_size'], self.image_size // model_configs['vision_cfg']['patch_size']
        idx = torch.arange(self.H*self.W, dtype=torch.float32, device=self.device)
        self.r_list = [1, 3]
        neighbor_conv_mask_scale_3 = F.unfold(F.pad(idx.reshape(1,1,self.H,self.W), (1,1,1,1), 'replicate'), (3,3)).squeeze().t().long()
        self.conv_masks = [None, neighbor_conv_mask_scale_3]
        self.GaussianBlur = transforms.GaussianBlur(kernel_size=int(2 * 3 * args.sigma + 1), sigma=4)
        
        self.PFM = [[torch.tensor([]).to(self.device, torch.float32) for _ in self.l_list] for _ in self.r_list]
        self.PSM = [[torch.tensor([]).to(self.device, torch.float32) for _ in self.l_list] for _ in self.r_list]
        self.score_memory = torch.tensor([]).to(self.device, torch.float32)
        self.AAIF_memory = [torch.tensor([]).to(self.device, torch.float32) for _ in self.l_list]
        self.IF_memory = torch.tensor([]).to(self.device, torch.float32)
        
        self.k_shot = args.k_shot
        self.normal_feature_num = [[0 for _ in self.l_list] for _ in self.r_list]
        if self.k_shot > 0:
            self.normal_feature_num_min = int(self.H * self.W * self.k_shot / 4)
            self.normal_weight = self.k_shot / 4
        
        self.tested_num = 0
        self.sample_num = int(args.keep_fratio * self.H * self.W)
        self.keep_snum = args.keep_snum
        self.keep_inum = args.keep_inum
        self.topk = args.topk
        self.LS_ratio = args.LS_ratio
        self.Rs = args.Rs
        self.Rs_freq = args.Rs_freq
        self.Rs_temp = args.Rs_temp
        self.max_Rs_num = args.max_Rs_num
        self.other = args.other

        if args.sampler == 'KCS':
            self.sample = self.KCS
        elif args.sampler == 'GCS':
            self.GCSer = GreedyCoresetSampler(args.keep_fratio, self.device)
            self.sample = self.GCS
        elif args.sampler == 'RS':
            self.sample = self.RS
        else:
            self.sample = self.SCS
        
    def renew_memory(self):
        self.PFM = [[torch.tensor([]).to(self.device, torch.float32) for _ in self.l_list] for _ in self.r_list]
        self.AAIF_memory = [torch.tensor([]).to(self.device, torch.float32) for _ in self.l_list]
        self.score_memory = torch.tensor([]).to(self.device, torch.float32)
        self.IF_memory = torch.tensor([]).to(self.device, torch.float32)
        self.normal_feature_num = [[0 for _ in self.l_list] for _ in self.r_list]
        self.foreground_map = None
        self.tested_num = 0
        
    def LS(self, sims, Y=0.02):
        sims /= torch.kthvalue(sims.flatten(), k=int(sims.flatten().shape[0] * (1 - Y)))[0]
        sims[sims > 1] = 1
        sims *= sims
        return sims
    
    def SCS(self, F_ref, S_ref=None, normal_fnum=0):
        if F_ref.shape[0] > self.sample_num:
            F_ref_sim = torch.tril(torch.mm(F_ref, F_ref.t()), diagonal=-1).max(dim=-1)[0]
            if self.k_shot and normal_fnum <= self.normal_feature_num_min:
                F_ref_sim[:normal_fnum] = 0
            _, min_idx = torch.topk(F_ref_sim, k=self.sample_num, largest=False, sorted=False)
            keep_idxs, _ = torch.sort(min_idx)
            F_ref = F_ref[keep_idxs, :].contiguous()
            if S_ref is not None:
                S_ref = S_ref[:, keep_idxs].contiguous()
            if self.k_shot and normal_fnum > self.normal_feature_num_min:
                normal_fnum = (keep_idxs < normal_fnum).sum()
        return F_ref, S_ref, normal_fnum
    
    def GCS(self, F_ref):
        keep_idxs = torch.tensor(self.GCSer._compute_greedy_coreset_indices(F, self.sample_num)).to(self.device).long()
        return F_ref[keep_idxs], None
    
    def RS(self, F_ref):
        keep_idxs = torch.randperm(F_ref.shape[0])[:self.sample_num]
        return F_ref[keep_idxs], None

    def KCS(self, X, max_iters=100, tol=1e-4):
        n_samples, n_features = X.shape
        k = self.sample_num
        indices = torch.randperm(n_samples, device=self.device)[:k]
        centers = X[indices]
        
        for _ in range(max_iters):
            distances = torch.cdist(X, centers)  # (n_samples, k)
            cluster_assignments = torch.argmin(distances, dim=1)  # (n_samples,)
            one_hot = F.one_hot(cluster_assignments, num_classes=k).float()  # (n_samples, k)
            counts = one_hot.sum(dim=0)  # (k,)
            new_centers = torch.mm(one_hot.T, X)  # (k, n_features)
            new_centers = new_centers / counts.unsqueeze(1).clamp(min=1e-10)
            mask = counts == 0
            new_centers = torch.where(mask.unsqueeze(1), centers, new_centers)
            if torch.norm(centers - new_centers, p='fro') < tol:
                break
                
            centers = new_centers
        
        assigned_centers = centers[cluster_assignments]
        point_distances = torch.norm(X - assigned_centers, p=2, dim=1)
        sampled_indices = []
        for i in range(k):
            mask = cluster_assignments == i
            if mask.any():
                cluster_distances = point_distances[mask]
                min_idx = torch.argmin(cluster_distances)
                original_indices = torch.where(mask)[0]
                sampled_indices.append(original_indices[min_idx])
            else:
                global_distances = torch.norm(X - centers[i], p=2, dim=1)
                sampled_indices.append(torch.argmin(global_distances))
        
        return X[torch.stack(sampled_indices)], None
    
    def segment_and_update(self, F, F_ref=None, S_ref=None, normal_fnum=0, update=True, sim_ratio=0.3):
        anomaly_map, k_shot_map = 0, 0
        if self.tested_num > 0:
            img_num, feat_num, _ = F_ref.shape
            patch2ref = torch.mm(F[0], F_ref.reshape(img_num * feat_num, -1).t()).reshape(-1, img_num, feat_num)
            max_sim, _ = torch.max(patch2ref, dim=-1)
            
            estimate_prob = torch.topk(max_sim, k=int(np.ceil(max_sim.shape[-1] * sim_ratio)), dim=-1, sorted=False)[0].mean(dim=-1)
            anomaly_map = (1 - estimate_prob).unsqueeze(0)
            
            if self.k_shot:
                k_shot_sim = patch2ref[:, :self.k_shot, :].max(dim=-1)[0].max(dim=-1)[0]
                k_shot_sim = self.LS(k_shot_sim)
                k_shot_map = (1 - k_shot_sim).unsqueeze(0)

            if update:
                if F_ref.shape[0] < self.keep_snum:
                    F_ref = torch.cat((F_ref, self.sample(F[0])[0].unsqueeze(0)), dim=0).contiguous()
                else:
                    F_ref = torch.cat((F_ref[:self.k_shot, :, :], F_ref[self.k_shot + 1:, :, :], self.sample(F[0])[0].unsqueeze(0)), dim=0).contiguous()
        else:
            F_ref = self.sample(F[0])[0].unsqueeze(0).contiguous()
        # print(F_ref.shape)
        return anomaly_map, k_shot_map, F_ref, S_ref, normal_fnum

    
    def process_image_and_update(self, input_image=None, update=True):
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_feature, patch_tokens = self.clip_model.encode_image(input_image, self.uni_features_list)
            patch_tokens = [p[:,1:,:] for p in patch_tokens]
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            if self.Rs and self.tested_num >= self.k_shot:
                self.IF_memory = torch.cat((self.IF_memory, image_feature), dim=0).contiguous()
                sim_mat = image_feature @ self.IF_memory.T
                if not update:
                    self.IF_memory = self.IF_memory[:-1]
                
            text_prob = (self.text_temp * image_feature @ self.text_features).softmax(dim=-1)[:, 1]
            anomaly_map_text = 0
            patch_tokens_linear = self.linear_layer([patch_tokens[i] for i in self.text_idx])
            for p in patch_tokens_linear:
                p /= p.norm(dim=-1, keepdim=True)
                anomaly_map_text += (self.text_temp * p @ self.text_features).softmax(dim=-1)[:, :, 1]
            anomaly_map_text /= len(self.text_idx)
            if update:
                self.foreground_map = (self.foreground_map * self.tested_num + anomaly_map_text[0]) / (self.tested_num + 1) if self.foreground_map is not None else anomaly_map_text[0]
            if self.foreground_map is not None:
                foreground_idx = torch.topk(self.foreground_map, k=int(self.foreground_map.shape[-1] * self.foreground_ratio), sorted=False)[1] if self.foreground_ratio < 1 else None
                anomaly_map_text = torch.cat((anomaly_map_text, self.foreground_map.unsqueeze(0)), dim=0).max(dim=0, keepdim=True)[0]
            else:
                foreground_idx = None
            

            anomaly_map_rare = 0
            k_shot_map = 0
            patch_features = torch.stack([patch_tokens[i] for i in self.rare_idx], dim=0)
            for ri, r in enumerate(self.r_list):
                if r > 1:
                    patch_features_r = patch_features[...,self.conv_masks[ri],:].mean(-2)
                else:
                    patch_features_r = patch_features.clone()

                patch_features_r /= patch_features_r.norm(dim=-1, keepdim=True)
                for l in self.l_list:
                    anomaly_map_rare_rl, k_shot_map_rl, self.PFM[ri][l], self.PSM[ri][l], self.normal_feature_num[ri][l] = self.segment_and_update(F=patch_features_r[l],
                                        F_ref=self.PFM[ri][l], S_ref=self.PSM[ri][l], normal_fnum=self.normal_feature_num[ri][l], update=update)
                    anomaly_map_rare += anomaly_map_rare_rl
                    k_shot_map += k_shot_map_rl

            foreground_score = 0
            k_shot_score = 0
            for l in self.l_list:
                if foreground_idx is not None:
                    aaif = patch_features[l,0][foreground_idx].mean(dim=0, keepdim=True)
                else:
                    aaif = patch_features[l,0].mean(dim=0, keepdim=True)
                aaif /= aaif.norm(dim=-1, keepdim=True)
                if self.tested_num > 0:
                    aaif_sim = torch.mm(aaif, self.AAIF_memory[l].t())
                    foreground_score += 1 - torch.topk(aaif_sim, k=int(np.ceil(aaif_sim.shape[-1] * 0.3)), sorted=False)[0].mean(-1)
                    if self.Rs and self.tested_num >= self.k_shot:
                        sim_mat += torch.cat((aaif_sim[:, self.k_shot:], torch.ones((1,1)).to(self.device)), dim=1)
                    if self.k_shot > 0:
                        k_shot_score += 1 - torch.max(aaif_sim[:,:self.k_shot], dim=-1)[0]
                if update:
                    self.AAIF_memory[l] = torch.cat((self.AAIF_memory[l], aaif), dim=0).contiguous()
                    
            if self.tested_num < self.k_shot:
                self.tested_num += 1
                # for ri, r in enumerate(self.r_list):
                #     for l in self.l_list:
                #         self.PFM[ri][l], self.PSM[ri][l], self.normal_feature_num[ri][l] = self.SCS(F_ref=self.PFM[ri][l], S_ref = self.PSM[ri][l], normal_fnum=self.normal_feature_num[ri][l])
                return 
            
            anomaly_map_rare /= len(self.l_list) * len(self.r_list)
            foreground_score /= len(self.l_list)
            if self.k_shot:
                k_shot_map /= len(self.l_list) * len(self.r_list)
                k_shot_score /= len(self.l_list)
                anomaly_map_rare = (anomaly_map_rare + k_shot_map * self.normal_weight) / (1 + self.normal_weight)
                foreground_score = (foreground_score + k_shot_score * self.normal_weight) / (1 + self.normal_weight)

            if foreground_score > 0:
                anomaly_map_rare += foreground_score * (1 + normalize01(self.foreground_map))
            
            if self.tested_num <= 0 or isinstance(anomaly_map_rare, float):
                anomaly_map = anomaly_map_text
            elif self.tested_num <= 3:
                anomaly_map = anomaly_map_rare * 2/3 + 2/3 * anomaly_map_text
            else:
                anomaly_map = anomaly_map_rare * 4/3 + 1/3 * anomaly_map_text

            # if self.tested_num == 0:
            #     anomaly_map = anomaly_map_text
            # elif self.tested_num <= 3:
            #     anomaly_map = anomaly_map_rare * 2/3 + 0.5 * anomaly_map_text
            # else:
            #     anomaly_map = anomaly_map_rare + 0.25 * anomaly_map_text
            # if foreground_score > 0:
            #     anomaly_map += foreground_score * (1 + normalize01(self.foreground_map))

            anomaly_map = F.interpolate(anomaly_map.view(-1, 1, self.H, self.W), size=self.image_size, mode='bilinear', align_corners=True)
            # anomaly_map = F.conv2d(anomaly_map, self.gaussian_kernel, padding=self.kernel_size//2)
            anomaly_map = self.GaussianBlur(anomaly_map)
            anomaly_score = anomaly_map.max() + text_prob
            
            if self.Rs:
                self.score_memory = torch.cat((self.score_memory, anomaly_score), dim=0).contiguous()
                if sim_mat.shape[-1] > self.Rs_freq:
                    Rs_num = min(int(np.ceil(sim_mat.shape[-1] / self.Rs_freq)), self.max_Rs_num + 1)
                    sim_mat /= sim_mat.max()
                    neighbor_sim, neighbor_idx = torch.topk(sim_mat, k=Rs_num, sorted=False)
                    anomaly_score = ((self.Rs_temp * neighbor_sim).softmax(dim=-1) * self.score_memory[neighbor_idx]).sum(dim=-1)
                if not update:
                    self.score_memory = self.score_memory[:-1]
                    
            if update:
                self.tested_num += 1
                # for ri, r in enumerate(self.r_list):
                #     for l in self.l_list:
                #         self.PFM[ri][l], self.PSM[ri][l], self.normal_feature_num[ri][l] = self.SCS(F_ref=self.PFM[ri][l], S_ref = self.PSM[ri][l], normal_fnum=self.normal_feature_num[ri][l])
            
            # if self.PSM[0][0].shape[0] > self.keep_snum:
            #     for ri, r in enumerate(self.r_list):
            #         for l in self.l_list:
            #             self.PSM[ri][l] = self.PSM[ri][l][1:]
                        
            if self.score_memory.shape[0] > self.keep_inum:
                self.score_memory = self.score_memory[1:]
                self.IF_memory = self.IF_memory[1:]
                for l in self.l_list:
                    self.AAIF_memory[l][self.k_shot:] = self.AAIF_memory[l][self.k_shot+1:]

            return anomaly_map, anomaly_score
        