
#Import the required libraries
import os
import torch
import torch.nn as nn
import functools
from UCC_net_factory import net_factory


# model_1 = nn.DataParallel(kaiming_normal_init_weight(net_factory(net_type='Unet', in_chns=3, class_num=4)))
# model_2 = nn.DataParallel(xavier_normal_init_weight(net_factory(net_type='Unet', in_chns=3, class_num=4)))

model = net_factory(net_type='unet_ucc', in_chns=3, class_num=4)
# model = nn.DataParallel(model)
# print(model)
