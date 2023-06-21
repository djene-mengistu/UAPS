
from UCC_unet import UNet, UNet_UCC


def net_factory(net_type="unet_ucc", in_chns=3, class_num=4):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    
    elif net_type == "unet_ucc":
        net = UNet_UCC(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net
