##################################################
# Author: {Cher Bass}
# Copyright: Copyright {2020}, {ICAM}
# License: {MIT license}
# Credits: {Hsin-Ying Lee}, {2019}, {https://github.com/HsinYingLee/MDMM}
##################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
ACTIVATION = nn.ReLU


class Identity(nn.Module):

    def forward(self, x):
        return x


def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)


def conv2d_bn_block(in_channels, out_channels, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block conv-bn-activation
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )


def deconv2d_block(in_channels, out_channels, use_upsample=False, kernel=4, stride=2, padding=1, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block deconv-activation
    NB: use_upsample = True helps to remove chessboard artifacts:
    https://distill.pub/2016/deconv-checkerboard/
    '''
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        activation(),
    )


def deconv2d_bn_block(in_channels, out_channels, use_upsample=False, kernel=4, stride=2, padding=1, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block deconv-bn-activation
    NB: use_upsample = True helps to remove chessboard artifacts:
    https://distill.pub/2016/deconv-checkerboard/
    '''
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )


def deconv3d_block(in_channels, out_channels, use_upsample=False, kernel=4, stride=2, padding=1, momentum=0.01, activation=ACTIVATION, output_padding=0):
    '''
    returns a block deconv-activation
    NB: use_upsample = True helps to remove chessboard artifacts:
    https://distill.pub/2016/deconv-checkerboard/
    '''
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride, padding=padding, output_padding=output_padding)
    return nn.Sequential(
        up,
        activation(),
    )


def deconv3d_bn_block(in_channels, out_channels, use_upsample=False, kernel=4, stride=2, padding=1, momentum=0.01, activation=ACTIVATION, output_padding=0):
    '''
    returns a block deconv-bn-activation
    NB: use_upsample = True helps to remove chessboard artifacts:
    https://distill.pub/2016/deconv-checkerboard/
    '''
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride, padding=padding, output_padding=output_padding)
    return nn.Sequential(
        up,
        nn.BatchNorm3d(out_channels, momentum=momentum),
        activation(),
    )


def dense_layer_bn(in_dim, out_dim, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block linear-bn-activation
    '''
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim, momentum=momentum),
        activation()
    )


def conv3d_bn_block(in_channels, out_channels, kernel=3, stride=1, padding=1, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block 3Dconv-3Dbn-activation
    '''
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        nn.BatchNorm3d(out_channels, momentum=momentum),
        activation(),
    )


def conv2d_block(in_channels, out_channels, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    '''
    returns a block conv-activation
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )


def deconv3x3_2d(in_planes, out_planes, use_upsample=False, stride=1):
    """3x3 deconvolution with padding"""
    if use_upsample:
        up = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False)
    else:
        up = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False)
    return up


def deconv3x3_2d_k3(in_planes, out_planes, use_upsample=False, stride=1):
    """3x3 deconvolution with padding"""
    if use_upsample:
        up = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False)
    else:
        up = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
    return up


def conv3x3_2d(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1_2d(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3d_block(in_channels, out_channels, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    '''
    returns a block 3D conv-activation
    '''
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )


def conv3x3_3d(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1_3d(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv3x3_3d(in_planes, out_planes, use_upsample=False, stride=1):
    """3x3 deconvolution with padding"""
    if use_upsample:
        up = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False)
    else:
        up = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False)
    return up


class conv_1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(conv_1x1, self).__init__()
        # conv1
        self.conv1 = conv1x1_2d(inplanes, planes)

    def forward(self, x):
        out = self.conv1(x)
        return out


class Transconv_up_2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, use_upsample=False, kernel=4):
        super(Transconv_up_2d, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        if kernel == 4:
            self.conv1 = deconv3x3_2d(inplanes, planes, use_upsample=use_upsample)
        else:
            self.conv1 = deconv3x3_2d_k3(inplanes, planes, use_upsample=use_upsample)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.LeakyReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2)
        self.stride = stride
        self.avpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.use_upsample = use_upsample
        self.replication_pad = nn.ReflectionPad2d(1) # same as ReflectionPad2d, but for 3d
        self.kernel = kernel

    def forward(self, x):
        if self.use_upsample:
            out = self.upsample(x + self.bias1a)
            out = self.replication_pad(out)
            out = self.conv1(out)
            out = self.relu(out + self.bias1b)
        else:
            out = self.conv1(x + self.bias1a)
            if self.kernel == 4:
                out = self.avpool(out + self.bias1b)
            out = self.relu(out)

        return out
