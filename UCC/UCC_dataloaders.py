import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm


from torchsummary import summary
# import segmentation_models_pytorch as smp

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_PATH = '/.../data/NEU_data/train_images/'
MASK_PATH = '/.../data/NEU_data/training_annot/'
IMAGE_PATH_test = '/.../data/NEU_data/test_images/'
MASK_PATH_test = '/.../data/NEU_data/test_annot/'

n_classes = 4

def create_df():
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

df = create_df()
print('Total Train Images: ', len(df))


def create_df_test():
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH_test):
        for filename in filenames:
            name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

df_test = create_df_test()
print('Total Test Images: ', len(df_test))
X_test = df_test['id'].values


#split data
XX_train, X_val = train_test_split(df['id'].values, test_size=0.15, random_state=69)
X_train, X_untrain = train_test_split(XX_train, test_size=0.9, random_state=45) #proportion of training data

print('Train Size   : ', len(X_train))
print('Unlabeled_Train Size   : ', len(X_untrain))
print('Val Size     : ', len(X_val))
print('Test Size    : ', len(X_test))

class NEUDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None:
            img = Image.fromarray(img)
        
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()
        
        
        return img, mask

class NEUDataset_SW(Dataset):
    
    def __init__(self, img_path, mask_path, X, mean, std, transform_1, transform_2, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.patches = patch
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
        
        if self.transform_1 is not None:
            aug_1 = self.transform_1(image=img, mask=mask)
            aug_2 = self.transform_2(image=img, mask=mask)
            img_1 = Image.fromarray(aug_1['image'])
            img_2 = Image.fromarray(aug_2['image'])
            mask_1 = aug_1['mask']
            mask_2 = aug_2['mask']
        
        if self.transform_1 is None:
            img_1 = Image.fromarray(img)
            img_2 = Image.fromarray(img)
        
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img_1 = t(img_1)
        img_2 = t(img_2)
        mask_1 = torch.from_numpy(mask_1).long()
        mask_2 = torch.from_numpy(mask_2).long()
        
        
        return img_1, img_2, mask_1, mask_2
    
#Defining the tranformation values
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

t_train = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(p=0.3), A.VerticalFlip(p=0.3), 
                     A.RandomBrightnessContrast((0,0.5),(0,0.5), p=0.3),
                     A.GridDistortion(p=0.2),
                    #  A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.3),
                     A.Blur(p = 0.3),
                     A.GaussNoise(p = 0.4)])  

t_untrain_weak = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_NEAREST),
                     A.RandomBrightnessContrast((0,0.5),(0,0.5), p=0.3),
                     A.Blur(p = 0.3),
                     A.GaussNoise(p = 0.4)
                    ]) 

t_untrain_strong = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_NEAREST), 
                    #  A.RandomBrightnessContrast((0,0.5),(0,0.5), p=0.3),
                     A.Blur(p = 0.3),
                     A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.3),
                    #  A.Equalize (mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.3),
                     A.PixelDropout (dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, always_apply=False, p=0.2),
                     A.GaussNoise(p = 0.4)])                     



t_val = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_NEAREST)])
t_test = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_NEAREST)])

#datasets
train_set = NEUDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train, patch=False)
unlabeled_train_set = NEUDataset_SW(IMAGE_PATH, MASK_PATH, X_untrain, mean, std, t_untrain_weak, t_untrain_strong, patch=False)
val_set = NEUDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val, patch=False)
test_set = NEUDataset(IMAGE_PATH_test, MASK_PATH_test, X_test, mean, std, t_test, patch=False)

#dataloader
batch_size= 16

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers = 8, pin_memory = True, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_train_set, batch_size=batch_size, num_workers = 8, pin_memory = True, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers = 8, pin_memory = True, shuffle=True)  
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers = 8, pin_memory = True,  shuffle=False)  


