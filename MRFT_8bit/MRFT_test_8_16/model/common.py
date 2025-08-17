import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class ImageDownsample(nn.Conv2d):
    def __init__(self, n_colors, scale):
        super(ImageDownsample, self).__init__(n_colors, n_colors, kernel_size=2*scale, bias = False, padding=scale//2,stride=scale)
        kernel_size=2*scale
        self.weight.data = torch.zeros(n_colors,n_colors,kernel_size,kernel_size)
        for i in range(n_colors):
            self.weight.data[i,i,:,:] = torch.ones(1,1,kernel_size,kernel_size)/(kernel_size*kernel_size)
        self.requires_grad = False

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)



class SimpleUpsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, bias=True):

        m = []
        m.append(conv(n_feat, scale*scale*3, 3, bias))
        m.append(nn.PixelShuffle(scale))
        super(SimpleUpsampler, self).__init__(*m)

class SimpleGrayUpsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, bias=True):

        m = []
        m.append(conv(n_feat, scale*scale, 3, bias))
        m.append(nn.PixelShuffle(scale))
        super(SimpleGrayUpsampler, self).__init__(*m)


class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)