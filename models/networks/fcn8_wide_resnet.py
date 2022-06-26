import torch.nn as nn
import torchvision
import torch
from skimage import morphology as morph
from src.modules.eprop import eprop
import torch.utils.model_zoo as model_zoo
from scripts.SEAM.network import resnet38_SEAM, resnet38_aff
import numpy as np
from torch import optim
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                      stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.silu = nn.SiLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.silu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes, output_stride, BatchNorm):
        super().__init__()
        if output_stride == 4:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 4, 6, 10]
        elif output_stride == 2:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        #self.aspp1 = _ASPPModule(inplanes, outplanes, 1, padding=0,dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, outplanes, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, outplanes, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, outplanes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                             nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False),
                                             #BatchNorm(outplanes),
                                             nn.SiLU(inplace=True))
        self.conv1 = nn.Conv2d(outplanes*4, outplanes, 1, bias=False)
        self.bn1 = BatchNorm(outplanes)
        self.silu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(0.0)
        self._init_weight()

    def forward(self, x):
        #x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)

        return self.dropout(x)
  
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class PASPP(nn.Module):
    def __init__(self, inplanes, outplanes, output_stride=4, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        if output_stride == 4:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 4, 6, 10]
        elif output_stride == 2:
            dilations = [1, 12, 24, 36]
        elif output_stride == 16:
            dilations = [1, 2, 3, 4]
        elif output_stride == 1:
            dilations = [1, 16, 32, 48]
        else:
            raise NotImplementedError
        self._norm_layer = BatchNorm
        self.silu = nn.SiLU(inplace=True)
        self.conv1 = self._make_layer(inplanes, inplanes // 4)
        self.conv2 = self._make_layer(inplanes, inplanes // 4)
        self.conv3 = self._make_layer(inplanes, inplanes // 4)
        self.conv4 = self._make_layer(inplanes, inplanes // 4)
        self.atrous_conv1 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[0], padding=dilations[0])
        self.atrous_conv2 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[1], padding=dilations[1])
        self.atrous_conv3 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[2], padding=dilations[2])
        self.atrous_conv4 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[3], padding=dilations[3])
        self.conv5 = self._make_layer(inplanes // 2, inplanes // 2)
        self.conv6 = self._make_layer(inplanes // 2, inplanes // 2)
        self.convout = self._make_layer(inplanes, inplanes)
    
    def _make_layer(self, inplanes, outplanes):
        layer = []
        layer.append(nn.Conv2d(inplanes, outplanes, kernel_size = 1))
        layer.append(self._norm_layer(outplanes))
        layer.append(self.silu)
        return nn.Sequential(*layer)
    
    def forward(self, X):
        print(X.shape)
        x1 = self.conv1(X)
        print(x1.shape)
        x2 = self.conv2(X)
        print(x2.shape)
        x3 = self.conv3(X)
        x4 = self.conv4(X)
        
        x12 = torch.add(x1, x2)
        x34 = torch.add(x3, x4)
        
        x1 = torch.add(self.atrous_conv1(x1),x12)
        x2 = torch.add(self.atrous_conv2(x2),x12)
        x3 = torch.add(self.atrous_conv3(x3),x34)
        x4 = torch.add(self.atrous_conv4(x4),x34)
        
        x12 = torch.cat([x1, x2], dim = 1)
        x34 = torch.cat([x3, x4], dim = 1)
        
        x12 = self.conv5(x12)
        x34 = self.conv5(x34)
        x = torch.cat([x12, x34], dim=1)
        x = self.convout(x)
        return x 

class FCN8(nn.Module):
    def __init__(self, n_classes, with_affinity=True, with_PASPP=True, with_affinity_average=False, shared=False, exp_dict=None):
        super().__init__()
        self.n_classes = n_classes
        self.shared = shared

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = torchvision.models.wide_resnet50_2(pretrained=True) #torchvision.models.wide_resnet50_2(pretrained=True)
        resnet_block_expansion_rate = resnet50_32s.layer1[0].expansion
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0)
        self.dropout_f6 = nn.Dropout()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0)
        self.dropout_f7 = nn.Dropout()

        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()

        self.resnet50_32s = resnet50_32s

        self.score_32s = nn.Conv2d(512 * resnet_block_expansion_rate,
                                   self.n_classes,
                                   kernel_size=1)

        self.score_16s = nn.Conv2d(256 * resnet_block_expansion_rate,
                                   self.n_classes,
                                   kernel_size=1)

        self.score_8s = nn.Conv2d(128 * resnet_block_expansion_rate,
                                  self.n_classes,
                                  kernel_size=1)
           
        self.scoring_layer = nn.Conv2d(4096, self.n_classes, kernel_size=1, stride=1, padding=0)
        self.upscore2 = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=16, stride=8, bias=False)

        # Initilize Weights
        self.scoring_layer.weight.data.zero_()
        self.scoring_layer.bias.data.zero_()
        
        self.score_pool3 = nn.Conv2d(256, self.n_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, self.n_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        self.upscore2.weight.data.copy_(get_upsampling_weight(self.n_classes, self.n_classes, 4))
        self.upscore_pool4.weight.data.copy_(get_upsampling_weight(self.n_classes, self.n_classes, 4))
        self.upscore8.weight.data.copy_(get_upsampling_weight(self.n_classes, self.n_classes, 16))
        self.eprop = eprop.EmbeddingPropagation()

        self.with_affinity = with_affinity
        self.with_PASPP = with_PASPP

        if with_PASPP:
            self.aspp = PASPP(2048,2048,4) #(2048,2048,4)
        if with_affinity or self.shared:
            self.model_aff = resnet38_aff.Net(self.n_classes, exp_dict).cuda()
            self.model_aff.load_state_dict(torch.load('/content/drive/MyDrive/deepfish/weight/resnet38_aff_SEAM.pth'), strict=False)

        self.with_affinity_average = with_affinity_average


        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False


    def extract_features(self, x_input):
        self.resnet50_32s.eval()
        x = self.resnet50_32s.conv1(x_input)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x) 
        x_8s = self.resnet50_32s.layer2(x)
        x_16s = self.resnet50_32s.layer3(x_8s)
        x_32s = self.resnet50_32s.layer4(x_16s)

        return x_8s, x_16s, x_32s

    def forward(self, x, return_cam=False, crf=False):
        n,c,h,w = x.size()
        self.resnet50_32s.eval()
        #print(self.resnet50_32s)
        input_spatial_dim = x.size()[2:]

        _x = self.resnet50_32s.conv1(x)
        #print('after conv: ', x.shape) #[1,64,256,256]
        _x = self.resnet50_32s.bn1(_x)
        _x = self.resnet50_32s.relu(_x)
        _x = self.resnet50_32s.maxpool(_x)

        l1 = self.resnet50_32s.layer1(_x)
        #print('after layer1: ', x.shape) #[1,256,128,128]

        l2 = self.resnet50_32s.layer2(l1)
        #print('after layer2: ', x.shape) #[1,512,64,64]
        logits_8s = self.score_8s(l2)

        l3 = self.resnet50_32s.layer3(l2)
        #print('after layer3: ', x.shape) #[1,1024,32,32]
        logits_16s = self.score_16s(l3)

        l4 = self.resnet50_32s.layer4(l3)      
        #print('after layer4: ', x.shape) #[1,2048,16,16]
        if self.with_PASPP:
          l4 = self.aspp(l4)

        #fc6 = self.dropout_f6(self.relu(self.fc6(l2)))
        #print('fc6: ', fc6.shape) 
        #fc7 = self.dropout_f7(self.relu(self.fc7(fc6)))
        #print('fc7: ', fc7.shape) 
        logits_32s = self.score_32s(l4)
        #print(logits_32s.shape) #[2,2,16,16]

        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]

        logits_16s += nn.functional.interpolate(logits_32s, size=logits_16s_spatial_dim, mode="bilinear", align_corners=True)
        #print('logits_16s: ', logits_16s.shape) #[1,2,32,32]
        logits_8s += nn.functional.interpolate(logits_16s, size=logits_8s_spatial_dim, mode="bilinear", align_corners=True)
        #print('logits_8s: ', logits_8s.shape) #[1,2,64,64]
        logits = nn.functional.interpolate(logits_8s, size=input_spatial_dim, mode="bilinear", align_corners=True)
        #print('logits: ', logits.shape) #[1,2,512,512]
        #print(logits_upsampled.shape)
        # SEMANTIC SEGMENTATION PART
        # first
        
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

# ===========================================================
# helpers
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

