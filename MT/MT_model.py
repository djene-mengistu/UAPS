#Import the required libraries
import os
import torch
import torch.nn as nn
import functools
from net_factory import net_factory

def create_model(ema=False):
    # Network definition
    model = net_factory(net_type='unet', in_chns=3, class_num=4)
    
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

model = create_model()
ema_model = create_model(ema=True)
