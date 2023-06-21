#The model for dual-segmetnation netwrok
#Import the required libraries
import os
import sys
sys.path.append('./')
import torch
import torch.nn as nn
import functools
from utilities.UAPS_net_factory import net_factory

model = net_factory(net_type='unet_ccps', in_chns=1, class_num=7)
model = nn.DataParallel(model)
# print(model)