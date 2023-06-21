
#Import the required libraries
import os
import torch
import torch.nn as nn
import functools
from net_factory import net_factory

model = net_factory(net_type='unet', in_chns=3, class_num=4)