 
#Import the required libraries
import os
from segmentation_models_pytorch.deeplabv3 import model
import torch
import torch.nn as nn
import functools
from net_factory import net_factory

model = net_factory(net_type='unet_cct', in_chns=3, class_num=4)
model = nn.DataParallel(model)