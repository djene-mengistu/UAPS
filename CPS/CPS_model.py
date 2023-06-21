#Import the required libraries
import os
import torch
import torch.nn as nn
import functools
from utilities.m_net_factory import net_factory
from utilities.model_initialization import kaiming_normal_init_weight, xavier_normal_init_weight, xavier_uniform_init_weight, sparse_init_weight


# model_1 = nn.DataParallel(kaiming_normal_init_weight(net_factory(net_type='Unet', in_chns=3, class_num=4)))
# model_2 = nn.DataParallel(xavier_normal_init_weight(net_factory(net_type='Unet', in_chns=3, class_num=4)))
# model_3 = nn.DataParallel(xavier_uniform_init_weight(net_factory(net_type='Unet', in_chns=3, class_num=4)))


model1 = net_factory(net_type='unet_f', in_chns=3, class_num=4)  
model2 = net_factory(net_type='unet_f', in_chns=3, class_num=4)  
# model3 = net_factory(net_type='unet_f', in_chns=3, class_num=4)
# model2 = xavier_normal_init_weight(model2)
# model3 = sparse_init_weight(model3)


model1 = nn.DataParallel(model1, device_ids=[0,1])
model2 = nn.DataParallel(model2, device_ids=[0,1])
# model3 = nn.DataParallel(model3, device_ids=[0,1])

# print(model1)