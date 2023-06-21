#The model for dual-segmetnation netwrok
#Import the required libraries
import os
from segmentation_models_pytorch.deeplabv3 import model
import torch
import torch.nn as nn
import functools
from utilities.UAPS_net_factory import net_factory
from utilities.model_initialization import kaiming_normal_init_weight, xavier_normal_init_weight, xavier_uniform_init_weight, sparse_init_weight
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used


# model_1 = nn.DataParallel(kaiming_normal_init_weight(net_factory(net_type='Unet', in_chns=3, class_num=4)))
# model_2 = nn.DataParallel(xavier_normal_init_weight(net_factory(net_type='Unet', in_chns=3, class_num=4)))

model = net_factory(net_type='unet_uaps', in_chns=3, class_num=6)
model = nn.DataParallel(model)
# print(model)