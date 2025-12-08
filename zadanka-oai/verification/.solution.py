import os
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
train_dir = Path("train")
class CLIPReID(nn.Module):
    def __init__(self, unfreeze_last_n_layers=1):
        super(CLIPReID, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # freeze all parameters firstly
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # unfreeze the last n transformer blocks
        total_blocks = len(self.clip_model.visual.transformer.resblocks)
        start_block = total_blocks - unfreeze_last_n_layers
        
        # unfreeze assigned layers
        for i in range(start_block, total_blocks):
            for param in self.clip_model.visual.transformer.resblocks[i].parameters():
                param.requires_grad = True
        
        # unfreeze the last layer
        for param in self.clip_model.visual.ln_post.parameters():
            param.requires_grad = True
            
    def get_trainable_params(self):
        """Return numbers and names of trainable parameters"""
        trainable_params = []
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
                total_params += param.numel()
        return trainable_params, total_params
    
    def forward_one(self, x):
        x = x.to(dtype=torch.float32, device=self.device)
        # for training mode, do not use torch.no_grad()
        if self.training:
            features = self.clip_model.encode_image(x)
        else:
            with torch.no_grad():
                features = self.clip_model.encode_image(x)
        features = features.to(dtype=torch.float32)
        return nn.functional.normalize(features, p=2, dim=1)
    
    def forward(self, x1, x2):
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        return feat1, feat2

    def preprocess(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]
class CropMatchDataset(Dataset):
    """Step one: match crop and orig"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # construct positive sample pairs with crop relationship
        self.pairs = []  # [(orig_img, crop_img), ...]
        
        print("train_dir:", os.listdir(os.path.join(data_dir, 'crop', 'male')))
        n = len(os.listdir(os.path.join(data_dir, 'crop', 'male')))
        print([(os.path.join(data_dir, 'orig', 'male', f'{i}.png'), os.path.join(data_dir, 'crop', 'male', f'{i}.png')) for i in range(1, n+1)])
        self.pairs = [(os.path.join(data_dir, 'orig', 'male', f'{i}.png'), os.path.join(data_dir, 'crop', 'male', f'{i}.png')) for i in range(1, n+1)] + \
                     [(os.path.join(data_dir, 'orig', 'female', f'{i}.png'), os.path.join(data_dir, 'crop', 'female', f'{i}.png')) for i in range(1, n+1)]
        print(self.pairs, type(self.pairs))
        print(f"Found {len(self.pairs)} matching pairs")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        orig_path, crop_path = self.pairs[idx]
        
        orig = Image.open(orig_path).convert('RGB')
        crop = Image.open(crop_path).convert('RGB')
        
        orig = self.preprocess(orig)
        crop = self.preprocess(crop)
        
        return orig, crop
    
def train_crop_matcher(train_dir, epochs=50, batch_size=32, unfreeze_last_n_layers=6):
    dataset = CropMatchDataset(train_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"Training set size: {len(dataset)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # assign number of layers to unfreeze when create model
    model = CLIPReID(unfreeze_last_n_layers=unfreeze_last_n_layers).to(device).float()
    
    # print trainable params info
    trainable_params, total_params = model.get_trainable_params()
    print(f"Number of trainable parameters: {total_params}")
    print("Trainable Layers:")
    for param_name in trainable_params:
        print(f"- {param_name}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.5, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    temp = 0.07
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for orig, crop in pbar:
            orig = orig.float().to(device)
            crop = crop.float().to(device)
            
            optimizer.zero_grad()
            
            orig_feat = F.normalize(model.forward_one(orig), dim=1)   # |v| = 1
            crop_feat = F.normalize(model.forward_one(crop), dim=1)
            
            # Info-NCE loss
            logits = (orig_feat @ crop_feat.t()) / temp    # (B, B)
            labels = torch.arange(orig_feat.size(0), device=device)
            loss_i2c = F.cross_entropy(logits, labels)  # orig→crop
            loss_c2i = F.cross_entropy(logits.t(), labels)  # crop→orig
            loss = 0.5 * (loss_i2c + loss_c2i)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.8f}'})
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.8f}')
        if avg_loss < best_loss:
            torch.save(model.state_dict(), 'crop_model.pth')
            best_loss = avg_loss
            print("Saved best model")
    
    return model
crop_model = train_crop_matcher(train_dir=train_dir)
class GenderMatchDataset(Dataset):
    """Step 2: match gender relations"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # construct positive sample pairs (same ID opposite gender)
        self.pairs = []  # [(male_img, female_img), ...]
        
        n = len(os.listdir(os.path.join(data_dir, 'crop', 'male')))
        self.pairs = [(os.path.join(data_dir, 'orig', 'male', f'{i}.png'), os.path.join(data_dir, 'orig', 'female', f'{i}.png')) for i in range(1, n+1)]
        
        print(f"Founded {len(self.pairs)} pairs of gender matching images")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        male_path, female_path = self.pairs[idx]
        
        male = Image.open(male_path).convert('RGB')
        female = Image.open(female_path).convert('RGB')
        
        male = self.preprocess(male)
        female = self.preprocess(female)
        
        return male, female

def train_gender_matcher(train_dir, epochs=50, batch_size=32, unfreeze_last_n_layers=6):
    dataset = GenderMatchDataset(train_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"Training set size: {len(dataset)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # assign number of layers to unfreeze when create model
    model = CLIPReID(unfreeze_last_n_layers=unfreeze_last_n_layers).to(device).float()
    
    # print info of trainable parameters
    trainable_params, total_params = model.get_trainable_params()
    print(f"Number of trainable parameters: {total_params}")
    print("Trainable layers:")
    for param_name in trainable_params:
        print(f"- {param_name}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.5, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    temp = 0.07
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for male, female in pbar:
            male = male.float().to(device)
            female = female.float().to(device)
            
            optimizer.zero_grad()
            
            male_feat = F.normalize(model.forward_one(male), dim=1)
            female_feat = F.normalize(model.forward_one(female), dim=1)
            
            # Info-NCE loss
            logits = (male_feat @ female_feat.t()) / temp
            labels = torch.arange(male_feat.size(0), device=device)
            loss_i2f = F.cross_entropy(logits, labels)
            loss_f2i = F.cross_entropy(logits.t(), labels)
            loss = 0.5 * (loss_i2f + loss_f2i)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.8f}'})
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.8f}')
        if avg_loss < best_loss:
            torch.save(model.state_dict(), 'gender_model.pth')
            best_loss = avg_loss
            print("Saved best model")
    
    return model 
gender_model = train_gender_matcher(train_dir=train_dir)
def extract_features(model, image_dir):
    features = {}
    for img_name in tqdm(os.listdir(image_dir)):
        if img_name.endswith('.png'):
            img_path = os.path.join(image_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = model.preprocess(img)
            img = img.unsqueeze(0).float().to(model.device)
            
            with torch.no_grad():
                feat = model.forward_one(img)
            features[img_name[:-4]] = feat.cpu().numpy()
    return features

def match_images(query_dir, gallery_dir, prevent_gallery_reuse, save_path):
    # Load two models
    crop_model = CLIPReID()
    gender_model = CLIPReID()
    crop_model.load_state_dict(torch.load('crop_model.pth'))
    gender_model.load_state_dict(torch.load('gender_model.pth'))
    
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    crop_model.to(device).eval()
    gender_model.to(device).eval()
    
    # Step1: find crop relations between query and gallery
    print("Extract features for finding crop relations...")
    query_features = extract_features(crop_model, query_dir)
    gallery_features = extract_features(crop_model, gallery_dir)
    
    # Store corresponding original images of each cropped query
    crop_matches = {}
    
    for q_name, q_feat in query_features.items():
        similarities = []
        for g_name, g_feat in gallery_features.items():
            sim = torch.nn.functional.cosine_similarity(
                torch.from_numpy(q_feat),
                torch.from_numpy(g_feat),
                dim=1
            ).item()
            similarities.append((g_name, sim))
        
        # Find the most similar gallery images (crop relation)
        similarities.sort(key=lambda x: x[1], reverse=True)
        crop_matches[q_name] = similarities[0][0]
    
    # Step2: Find gender matching in gallery
    print("Extract features for gender matching...")
    gallery_gender_features = extract_features(gender_model, gallery_dir)
    
    n = len(os.listdir(query_dir))
    results = np.zeros(n)
    matched_galleries = set()  # for tracking matched images
    print("Matching images...")
    
    for q_name, matched_crop in tqdm(crop_matches.items()):
        # get features of matched_crop
        crop_feat = gallery_gender_features[matched_crop]
        
        # Find the most similar one in gallary, but exclude crop relation and matched images
        similarities = []
        for g_name, g_feat in gallery_gender_features.items():
            if g_name != matched_crop:
                if not prevent_gallery_reuse or g_name not in matched_galleries:
                    sim = torch.nn.functional.cosine_similarity(
                        torch.from_numpy(crop_feat),
                        torch.from_numpy(g_feat),
                        dim=1
                    ).item()
                    similarities.append((g_name, sim))
        
        # Find the most similar one in the remaining images
        if similarities:  # make sure candidates are not empty
            similarities.sort(key=lambda x: x[1], reverse=True)
            gender_match = similarities[0][0]
            if prevent_gallery_reuse:
                matched_galleries.add(gender_match)  # add the matched image into set
            # print(q_name, gender_match)
            results[int(q_name)-1] = int(gender_match)
    
    # save the results
    np.save(save_path, results)
    print(f"Matched {query_dir} with {gallery_dir}, results saved to {save_path}")