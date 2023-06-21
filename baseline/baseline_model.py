#Import the required libraries
import os
import sys
sys.path.append('./')
import torch
import torch.nn as nn
import functools
from baseline_net_factory import net_factory
# from segmentation_models_pytorch import DeepLabV3Plus
# from UperNet.upernet import get_upernet

def create_model(ema=False):
    # Network definition
    model = net_factory(net_type='unet', in_chns=3, class_num=4)
    
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

model1 = create_model()
# model1 = get_upernet() #When using UpeNet as a baseline
# model1 = DeepLabV3Plus("mobilenet_v2", encoder_weights= None, classes=4, activation=None) #When using DLV3+ as a baseline
# print (model1)