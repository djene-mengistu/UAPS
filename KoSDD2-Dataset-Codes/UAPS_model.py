#The model for dual-segmetnation netwrok
#Import the required libraries
import os
import torch
import torch.nn as nn
import functools
from utilities.UAPS_net_factory import net_factory

model = net_factory(net_type='unet_uaps', in_chns=3, class_num=2)

model = nn.DataParallel(model)