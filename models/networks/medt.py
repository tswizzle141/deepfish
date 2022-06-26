import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
from src.modules.eprop import eprop
import torch.utils.model_zoo as model_zoo
from scripts.SEAM.network import resnet38_SEAM, resnet38_aff
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=2, dilation=2, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding=dilation, dilation=dilation,bias=bias)
        self.bn = nn.BatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.silu = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        #x = self.silu(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class qkv_transform(nn.Conv1d):
  """Conv1d for qkv_transform"""

class AxialAttention(nn.Module):
    
    def __init__(self,
               in_planes,
               out_planes,
               groups=8,
               kernel_size=56,
               stride=1,
               bias=False,
               width=False,
               sep=False,
               ):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias = False) # because Batchnorm
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.rand(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            if sep :
                self.pooling = SeparableConv2d(out_planes, out_planes)
            else:
                self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else :
            x = x.permute(0, 3, 1, 2)
        N, W, C, H = x.shape
        x = x.contiguous().view(N*W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N*W, self.groups, self.group_planes*2, H), 
                              [self.group_planes // 2,self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes*2, self.kernel_size, self.kernel_size )
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2,self.group_planes // 2, self.group_planes],
                                                            dim = 0)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgcj,cij->bgij',k, k_embedding)
        qk = torch.einsum('bgci, bgcj -> bgij',q , k)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        # bn_similarity chanels dim = self.group * 3
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N*W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum("bgij,bgci->bgci", similarity, v)
        sve = torch.einsum("bgij,cij->bgci", similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N*W, self.out_planes*2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)
        if self.stride > 1:
            output = self.pooling(output)
        return output
  
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0, math.sqrt(1. / self.group_planes))

class AxialAttention_dynamic(nn.Module):
    def __init__(self,
               in_planes,
               out_planes,
               groups=8,
               kernel_size=56,
               stride=1,
               bias=False,
               width=False,
               sep=False,):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width
        # Priority on encoding

            ## Initial values 

        self.f_qr = nn.Parameter(torch.tensor(0.1),  requires_grad=False) 
        self.f_kr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0),  requires_grad=False)
        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias = False) # because Batchnorm
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.rand(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            if sep :
                self.pooling = SeparableConv2d(out_planes, out_planes)
            else :
                self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else :
            x = x.permute(0, 3, 1, 2)
        N, W, C, H = x.shape
        x = x.contiguous().view(N*W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N*W, self.groups, self.group_planes*2, H), 
                              [self.group_planes//2,self.group_planes//2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes*2, self.kernel_size, self.kernel_size )
        #print(all_embeddings.shape) #[4,32,32]
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2,self.group_planes // 2, self.group_planes],
                                                            dim = 0)

        #print('q: ', q.shape) #[128,8,1,128]
        #print('q_embedding: ', q_embedding.shape) #[1,32,32]

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding) #bgci
        kr = torch.einsum('bgcj,cij->bgij',k, k_embedding) #bgcj
        qk = torch.einsum('bgci, bgcj -> bgij',q , k)

        #print('qr: ', qr.shape)
        #print('kr: ', kr.shape)
        #print('qk: ', kr.shape) #[128,8,32,32]

        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)
        #print('qr: ', qr.shape)
        #print('kr: ', kr.shape)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        # bn_similarity chanels dim = self.group * 3
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N*W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum("bgij,bgci->bgci", similarity, v)
        sve = torch.einsum("bgij,cij->bgci", similarity, v_embedding)

        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N*W, self.out_planes*2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)
        if self.stride > 1:
            output = self.pooling(output)
        return output
  
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0, math.sqrt(1. / self.group_planes))

class AxialAttention_wopos(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                stride=1, bias=False, width=False, sep=False,):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_wopos, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                            padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups )

        self.bn_output = nn.BatchNorm1d(out_planes * 1)

        if stride > 1:
            if sep:
                self.pooling = SeparableConv2d(out_planes, out_planes)
            else:
                self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = self.bn_similarity(qk).reshape(N * W, 1, self.groups, H, H).sum(dim=1).contiguous()

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        sv = sv.reshape(N*W,self.out_planes * 1, H).contiguous()
        output = self.bn_output(sv).reshape(N, W, self.out_planes, 1, H).sum(dim=-2).contiguous()


        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))

class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self,
               in_planes,
               planes,
               stride=1,
               downsample=None,
               groups=1,
               base_width=64,
               dilation=1,
               norm_layer=None,
               kernel_size=56,
               sep=False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True,sep=sep)
        self.conv_up = conv1x1(width, planes*self.expansion)
        self.bn2 = norm_layer(planes*self.expansion)
        self.silu = nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.silu(out)
        print(out.shape)
        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.silu(out)
        return out

class AxialBlock_dynamic(nn.Module):

    expansion = 2

    def __init__(self,
               in_planes,
               planes,
               stride=1,
               downsample=None,
               groups=1,
               base_width=64,
               dilation=1,
               norm_layer=None,
               kernel_size=56,
               sep=False,):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True,sep=sep)
        self.conv_up = conv1x1(width, planes*self.expansion)
        self.bn2 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AxialBlock_wopos(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                base_width=64, dilation=1, norm_layer=None, kernel_size=56,sep=False):
        
        super(AxialBlock_wopos, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # print(kernel_size)
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.conv1 = nn.Conv2d(width, width, kernel_size = 1)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True,sep=sep)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.silu = nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # pdb.set_trace()

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.silu(out)
        # print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)

        out = self.silu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.silu(out)

        return out

class medt_net(nn.Module):

    def __init__(self, block, block_2, layers, num_classes=2, zero_init_residual=True,
               groups=8, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
               s=0.125, img_size=128, imgchan=3, sep=True, with_affinity=False, 
               with_affinity_average=False, shared=False, exp_dict=None):
        super().__init__()
        self.n_class = num_classes
        self.shared = shared
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.sep = sep
        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                                 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) #7
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.silu = nn.SiLU(inplace=True)
        img_size = img_size // 2

        # Global branch
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size // 2))
        self.layer2 = self._make_layer(block, int(256*s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=replace_stride_with_dilation[0])
        #self.aspp = PASPP(int(512*s), int(512*s), 2, self._norm_layer)
        
        #self.decoder4 = nn.Conv2d(int(512*s), int(256*s), kernel_size=3, stride=1, padding=1)
        #self.decoder5 = nn.Conv2d(int(256*s), int(128*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = self._make_decoder(int(512*s), int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = self._make_decoder(int(256*s), int(128*s), kernel_size=3, stride=1, padding=1)
        self.decoder6 = self._make_decoder(int(128*s), int(128*s), kernel_size=3, stride=1, padding=1)
        #self.aspp_out = ASPP(int(128*s), int(128*s), 2, self._norm_layer)
        
        self.adjust = nn.Conv2d(int(128*s), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)

        # Local branch  
        self.conv1_p = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) #7
        self.conv2_p = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_p = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1_p = norm_layer(self.inplanes)
        self.bn2_p = norm_layer(128)
        self.bn3_p = norm_layer(self.inplanes)

        self.relu_p = nn.SiLU(inplace=True)

        img_size_p = 256 // 4

        self.layer1_p = self._make_layer(block, int(128 * s), layers[0], kernel_size = (img_size_p//2))
        self.layer2_p = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size_p//2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3_p = self._make_layer(block, int(512*s), layers[2], stride=2, kernel_size=(img_size_p//4),
                                       dilate=replace_stride_with_dilation[1])
        #self.aspp2 = PASPP(int(1024*s), int(1024*s), 16, self._norm_layer)
        
        self.layer4_p = self._make_layer(block, int(1024*s), layers[3], stride=2, kernel_size=(img_size_p//8),
                                       dilate=replace_stride_with_dilation[2])
        
        """"
        self.decoder1_p = nn.Conv2d(int(1024 * 2 * s), int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2_p = nn.Conv2d(int(1024 * 2 * s),int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder3_p = nn.Conv2d(int(1024*s), int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4_p = nn.Conv2d(int(512*s), int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5_p = nn.Conv2d(int(256*s), int(128*s), kernel_size=3, stride=1, padding=1)
        self.decoderf = nn.Conv2d(int(128*s), int(128*s), kernel_size=3, stride=1, padding=1)
        """
        self.decoder1_p = nn.Conv2d(int(1024 * 2 * s), int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2_p = self._make_decoder(int(1024 * s),int(1024*s), kernel_size=3, stride=2, padding=1)
        self.decoder3_p = self._make_decoder(int(1024*s), int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4_p = self._make_decoder(int(512*s), int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5_p = self._make_decoder(int(256*s), int(128*s), kernel_size=3, stride=1, padding=1)
        self.decoderf = self._make_decoder(int(128*s), int(128*s), kernel_size=3, stride=1, padding=1)
        
        #self.aspp_out = PASPP(int(128*s), int(128*s), 1, self._norm_layer)
        self.adjust_p = nn.Conv2d(int(128*s), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft_p = nn.Softmax(dim=1)

        self.with_affinity = with_affinity

        if with_affinity or self.shared:
            self.model_aff = resnet38_aff.Net(self.n_class, exp_dict).cuda()
            self.model_aff.load_state_dict(torch.load('/content/drive/MyDrive/deepfish/weight/resnet38_aff_SEAM.pth'), strict=False)

        self.with_affinity_average = with_affinity_average

        self._init_weight()
    
    def _make_decoder(self, inplanes, outplanes, kernel_size=3, stride=1, padding=1):
        layer = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.SiLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.Dropout(p=0.1), # save load thi bo Dropout
            nn.BatchNorm2d(outplanes),
            nn.SiLU(inplace=True),
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1))
        return layer
        

    def _make_layer(self,
                  block,
                  planes,
                  blocks,
                  kernel_size=56,
                  stride=1,
                  dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.sep :
                downsample = SeparableConv2d(self.inplanes, planes*block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ) 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width = self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size, sep=self.sep))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                             base_width=self.base_width, dilation=self.dilation,
                             norm_layer=norm_layer, kernel_size=kernel_size))
        return nn.Sequential(*layers)
  
  
  
    def _forward_impl(self, x, return_cam=False, crf=False):
        xin = x.clone()

        # Global branch
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        #print(x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.silu(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        #x1_aspp = self.aspp(x)
        #x2 = self.aspp(x2)
        x = F.silu(F.interpolate(self.decoder4(x2), scale_factor=(2,2), mode='bilinear'))
        x = torch.add(x, x1)
        x = F.silu(F.interpolate(self.decoder5(x), scale_factor=(2,2), mode='bilinear'))
        x = F.silu(F.interpolate(self.decoder6(x), scale_factor=(2,2), mode='bilinear'))
        
        #x = self.aspp_out(x)

        # Local branch
        x_loc = x.clone()
        for i in range(0,4):
            for j in range(0,4):
                x_p = xin[:,:,64*i:64*(i+1),64*j:64*(j+1)]
                x_p = self.conv1_p(x_p)
                x_p = self.bn1_p(x_p)
                x_p = self.conv2_p(x_p)
                x_p = self.bn2_p(x_p)
                x_p = self.conv3_p(x_p)
                x_p = self.bn3_p(x_p)

                x_p = self.relu(x_p)

                x1_p = self.layer1_p(x_p)
                x2_p = self.layer2_p(x1_p)
                x3_p = self.layer3_p(x2_p)
                #x3_p = self.aspp2(x3_p)
                #x4_p = self.layer4_p(x3_p)

                #x_p = F.silu(F.interpolate(self.decoder1_p(x4_p), scale_factor=(2,2), mode='bilinear'))
                #x_p = torch.add(x_p, x4_p)
                x_p = F.silu(F.interpolate(self.decoder2_p(x3_p), scale_factor=(2,2), mode='bilinear'))
                x_p = torch.add(x_p, x3_p)
                x_p = F.silu(F.interpolate(self.decoder3_p(x_p), scale_factor=(2,2), mode='bilinear'))
                x_p = torch.add(x_p, x2_p)
                x_p = F.silu(F.interpolate(self.decoder4_p(x_p), scale_factor=(2,2), mode='bilinear'))
                x_p = torch.add(x_p, x1_p)
                x_p = F.silu(F.interpolate(self.decoder5_p(x_p), scale_factor=(2,2), mode='bilinear'))

                x_loc[:,:,64*i:64*(i+1),64*j:64*(j+1)] = x_p

        x = torch.add(x, x_loc)
        #x = self.aspp_out(x)
        x = F.silu(self.decoderf(x))
        logits = self.adjust(F.relu(x))
        if self.shared:
            logits = cam = self.model_aff.output_logits(x)
        if self.with_affinity:
            logits_aff = self.model_aff.apply_affinity(x, logits, crf=crf)

            if self.with_affinity_average:
                logits = (logits_aff + logits) / 2.
            else:
                logits = logits_aff

        #if return_features:
            #return logits, upscore_pool4, x

        if return_cam:
            return cam, logits_aff
        return logits
  
    def forward(self, x):
        return self._forward_impl(x)
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def MedT(pretrained=False, **kwargs):
    model = medt_net(AxialBlock_dynamic,AxialBlock_wopos, [1, 2, 4, 1], s= 0.125,  **kwargs)
    return model