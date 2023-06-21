
from utilities.baseline_unet import UNet


def net_factory(net_type="unet_cct", in_chns=3, class_num=4):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net
