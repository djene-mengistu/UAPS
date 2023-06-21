
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import copy

from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms.functional as transforms_f


# --------------------------------------------------------------------------------
# Define useful functions
# --------------------------------------------------------------------------------
def label_binariser(inputs):
    outputs = torch.zeros_like(inputs).to(inputs.device)
    index = torch.max(inputs, dim=1)[1]
    outputs = outputs.scatter_(1, index.unsqueeze(1), 1.0)
    return outputs


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
    # we will still mask out those invalid values in valid mask
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)


def denormalise(x, imagenet=True):
    if imagenet:
        x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2


def create_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def tensor_to_pil(im, label, logits):
    im = denormalise(im)
    im = transforms_f.to_pil_image(im.cpu())

    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())

    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
    return im, label, logits


# --------------------------------------------------------------------------------
# Define semi-supervised methods (based on data augmentation)
# --------------------------------------------------------------------------------
def generate_cutout_mask(img_size, ratio=2): #ratio=random.choice([2,3,4,5])
    
    cutout_area = img_size[0] * img_size[1] / ratio

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.float()


def generate_mix_data(data, target, mode='cutmix', p=0.2): #For mixing in the same loader (within labeled samples)
    batch_size, _, im_h, im_w = data.shape
    device = data.device

    new_data = []
    new_target = []
    # new_logits = []
    for i in range(batch_size):

        if mode == 'cutmix':
            mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
        if random.random() < p:            
            new_data.append((data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
            new_target.append((target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
            # new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        else: 
            new_data.append((data[i].unsqueeze(0)))
            new_target.append((target[i]).unsqueeze(0))
            # new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

    new_data, new_target = torch.cat(new_data), torch.cat(new_target)
    return new_data, new_target.long()

def generate_crossmix_data(data_l, data_wk, data_st, mode='cutmix', p = 0.3):
    batch_size, _, im_h, im_w = data_l.shape
    device = data_l.device

    new_data_wk = []
    new_data_st = []
    # new_target = []
    # new_logits = []
    for i in range(batch_size):
        
        if mode == 'cutmix':
            mix_mask = generate_cutout_mask([im_h, im_w]).to(device)

        if random.random() < p:
            new_data_wk.append((data_wk[i] * mix_mask + data_l[i] * (1 - mix_mask)).unsqueeze(0))
            new_data_st.append((data_st[i] * mix_mask + data_l[i] * (1 - mix_mask)).unsqueeze(0))
        else:
            new_data_wk.append((data_wk[i]).unsqueeze(0))
            new_data_st.append((data_st[i]).unsqueeze(0))
        # new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

    new_data_wk, new_data_st = torch.cat(new_data_wk), torch.cat(new_data_st)
    return new_data_wk, new_data_st



