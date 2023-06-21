#The model for dual-segmetnation netwrok
#Import the required libraries
import os
import torch
import torch.nn as nn
import functools
from utilities.UAPS_net_factory import net_factory
# from utilities.model_initialization import kaiming_normal_init_weight, xavier_normal_init_weight, xavier_uniform_init_weight, sparse_init_weight



model = net_factory(net_type='unet_uaps', in_chns=3, class_num=4)
model = nn.DataParallel(model) #Training the models on parallel GPU
# print(model)