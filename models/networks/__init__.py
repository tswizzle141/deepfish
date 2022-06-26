from . import fcn8_vgg16, unet, unet_resnet, attention_unet, fcn8_resnet, fcn8
from . import fcn8_wide_resnet, res2net, medt, transunet, swinunet
from torchvision import models
import torch, os
import torch.nn as nn


def get_network(network_name, n_classes, exp_dict):
    #if network_name == 'fcn8_vgg16_att':
        #model_base = fcn8_vgg16.FCN8VGG16(n_classes=n_classes, with_attention=True)

    if network_name == 'fcn8_vgg16':
        model_base = fcn8_vgg16.FCN8VGG16(n_classes=n_classes, 
                    with_attention=exp_dict['model'].get('with_attention'),
                    with_affinity=exp_dict['model'].get('with_affinity'),
                    with_affinity_average=exp_dict['model'].get('with_affinity_average'),
                    shared=exp_dict['model'].get('shared'),
                    exp_dict=exp_dict)

    elif network_name == "unet_resnet":
        model_base = unet_resnet.ResNetUNet(n_class=n_classes)

    elif network_name == "unet":
        model_base = unet.UNet(n_classes=n_classes, n_channels=3)

    elif network_name == "swinunet":
        model_base = swinunet.SwinUnet()

    elif network_name == "transunet":
        model_base = transunet.TransUNet(img_dim=512,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=2)

    elif network_name == "medt":
        #model_base = medt.MedT(medt.AxialBlock_dynamic,medt.AxialBlock_wopos, [1, 2, 4, 1], s= 0.125)
        model_base = medt.MedT(img_size = 512)

    elif network_name == "attention_unet":
        model_base = attention_unet.AttU_Net()

    #elif network_name == "res2net":
        #model_base = res2net.Res2Net(block=res2net.Bottle2neck, layers=[3, 4, 6, 3])

    elif network_name == "fcn8":
        return fcn8.FCN8(n_class=n_classes)    

    elif network_name == "fcn8_resnet50":
        return fcn8_resnet.FCN8(n_classes)
        
    elif network_name == "fcn8_wide_resnet50":
        return fcn8_wide_resnet.FCN8(n_classes, 
                    with_affinity=exp_dict['model'].get('with_affinity'),
                    with_affinity_average=exp_dict['model'].get('with_affinity_average'),
                    shared=exp_dict['model'].get('shared'),
                    exp_dict=exp_dict)
    return model_base

