import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm

from .common import BaseNetwork
from  .resnet import Resnet18
#from carafe import CARAFEPack
#from mmcv.ops.carafe import CARAFEPack #train

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False,
                 bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)
            
class RPPModule(nn.Module):
    def __init__(self, in_channels: int, groups=2) -> None:
        super(RPPModule, self).__init__()
        self.groups = groups
        self.conv_dws1 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=4,
                                    group=1, dilation=4, bn_act=True))
        self.conv_dws2 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=8,
                                    group=1, dilation=8, bn_act=True))

        self.fusion = nn.Sequential(
            conv_block(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True))

        self.conv_dws3 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        br1 = self.conv_dws1(x)
        b2 = self.conv_dws2(x)

        out = torch.cat((br1, b2), dim=1)
        out = self.fusion(out)
        br3 = self.conv_dws3(F.adaptive_avg_pool2d(x, (1, 1)))
        output = br3 + out

        return output

##rpp improve
class RPPModule0(nn.Module):
    def __init__(self, in_channels: int, groups=2) -> None:
        super(RPPModule0, self).__init__()
        self.groups = 2
        self.conv_dws1 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                    group=1, dilation=1, bn_act=True))
        self.conv_dws2 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=2,
                                    group=1, dilation=2, bn_act=True))

        self.fusion = nn.Sequential(
            conv_block(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True))

        self.conv_dws3 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3, padding=0, dilation=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        br1 = self.conv_dws1(x)
        b2 = self.conv_dws2(x)

        out = torch.cat((br1, b2), dim=1)
        out = self.fusion(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        output = x * (1 - mask) + out * mask
        return output

class RPPModule1(nn.Module):
    def __init__(self, in_channels: int, groups=2) -> None:
        super(RPPModule1, self).__init__()
        self.groups = groups
        self.conv_dws1 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=2,
                                    group=1, dilation=2, bn_act=True))
        self.conv_dws2 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=4,
                                    group=1, dilation=4, bn_act=True))

        self.fusion = nn.Sequential(
            conv_block(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True))

        self.conv_dws3 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3, padding=0, dilation=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        br1 = self.conv_dws1(x)
        b2 = self.conv_dws2(x)

        out = torch.cat((br1, b2), dim=1)
        out = self.fusion(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        output = x * (1 - mask) + out * mask
        return output
        
class RPPModule2(nn.Module):
    def __init__(self, in_channels: int, groups=2) -> None:
        super(RPPModule2, self).__init__()
        self.groups = groups
        self.conv_dws1 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=4,
                                    group=1, dilation=4, bn_act=True))
        self.conv_dws2 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=8,
                                    group=1, dilation=8, bn_act=True))

        self.fusion = nn.Sequential(
            conv_block(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True))

        self.conv_dws3 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3, padding=0, dilation=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        br1 = self.conv_dws1(x)
        b2 = self.conv_dws2(x)

        out = torch.cat((br1, b2), dim=1)
        out = self.fusion(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        output = x * (1 - mask) + out * mask
        return output
        
class RPPModule3(nn.Module):
    def __init__(self, in_channels: int, groups=2) -> None:
        super(RPPModule3, self).__init__()
        self.groups = groups
        self.conv_dws1 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=8,
                                    group=1, dilation=8, bn_act=True))
        self.conv_dws2 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=16,
                                    group=1, dilation=16, bn_act=True))

        self.fusion = nn.Sequential(
            conv_block(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True))

        self.conv_dws3 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3, padding=0, dilation=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        br1 = self.conv_dws1(x)
        b2 = self.conv_dws2(x)

        out = torch.cat((br1, b2), dim=1)
        out = self.fusion(out)
        mask1 = self.gate(x)#[8, 64, 40, 40]
        mask = my_layer_norm(mask1)#[8, 64, 40, 40]
        mask = torch.sigmoid(mask)#[8, 64, 40, 40]
        output = x * (1 - mask) + out * mask#[8, 64, 40, 40]
        return output
        
class SPFM(nn.Module):
    def __init__(self, in_channels, out_channels, num_splits):
        super(SPFM,self).__init__()

        assert in_channels % num_splits == 0

        self.in_channels = 256
        self.out_channels = 256
        self.num_splits = 4
        self.subspaces = nn.ModuleList(
            [RPPModule0(int(self.in_channels / self.num_splits)),
            RPPModule1(int(self.in_channels / self.num_splits)),
            RPPModule2(int(self.in_channels / self.num_splits)),
            RPPModule3(int(self.in_channels / self.num_splits))]
            )

        self.out = conv_block(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)

    def forward(self, x):
        group_size = int(self.in_channels / self.num_splits)
        #print(x.shape)#8,256,40,40
        sub_Feat = torch.chunk(x, self.num_splits, dim=1)#yuan zu
        out = []
        for id, l in enumerate(self.subspaces):
            out.append(self.subspaces[id](sub_Feat[id]))
        out = torch.cat(out, dim=1)#[8, 256, 40, 40]
        #print(out.shape)
        out = self.out(out)#[8, 256, 40, 40]
        #print(out.shape)
        return out

class InpaintGenerator(BaseNetwork):
    def __init__(self, args):  # 1046
        super(InpaintGenerator, self).__init__()
        self.conv1d = nn.Conv2d(512, 256, 1, stride=1, padding=0)
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7), #4 
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.middle = nn.Sequential(*[AOTBlock(256, args.rates) for _ in range(args.block_num)])#256
        #self.middle = nn.Sequential(*[SPFM(256, 256,4) for _ in range(args.block_num)]
        #)
        self.spfm  = SPFM(256,256,4)

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            #UpModule(256,2),
            #nn.Conv2d(256, 128, 3, stride=1, padding=1),
            #nn.Conv2d(256, 128, 1, stride=1, padding=0),
            nn.ReLU(True),
            UpConv(128, 64),
            #UpModule(128,2),
            #nn.Conv2d(128, 64, 3, stride=1, padding=1),
            #nn.Conv2d(128, 64, 1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x, mask, feat_fuse):
        x = torch.cat([x, mask], dim=1) #torch.cat([x, mask], dim=1)
        x = self.encoder(x)#[4, 256, 64, 64])
        #print("x1",x.size())
        feat_fuse = F.interpolate(feat_fuse, scale_factor=2, mode="bilinear",align_corners=True)#4, 256, 64, 64]
        #print('feat_fuse1',feat_fuse.size()) 
        x = torch.cat([x,feat_fuse],dim=1)#[4, 512, 64, 64])
        #print("x2",x.size())
        x = self.conv1d(x)#[4, 256, 64, 64]
        #print("x3",x.size())
        x = self.middle(x)
       # x=x.ToTensor()
        #print(x.shape)  #8,256,40,40,
        # x =  self.spfm(x)
        #print(x.shape)
        # x =  self.spfm(x)
        # x =  self.spfm(x)
        # x =  self.spfm(x)
        # x =  self.spfm(x)
        # x =  self.spfm(x)
        # x =  self.spfm(x)
        # x =  self.spfm(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))

class UpModule(nn.Module):
    def __init__(self, in_channel, m):
       super(UpModule, self).__init__()
       self.model = CARAFEPack(channels=in_channel,scale_factor=m)
    def forward(self, x):
       x = self.model(x)
       return x


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)#average
    std = feat.std((2, 3), keepdim=True) + 1e-9#fangcha
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat###normalize




# ----- discriminator -----
class Discriminator(BaseNetwork):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        inc = 3
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x):
        feat = self.conv(x)
        return feat
        
#################################


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)                                                      

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


### This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        ## here self.sp is deleted
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)  # here return res3b1 feature
        feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_fuse#feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
