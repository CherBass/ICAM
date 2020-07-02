##################################################
# Author: {Cher Bass}
# Copyright: Copyright {2020}, {ICAM}
# License: {MIT license}
# Credits: {Hsin-Ying Lee}, {2019}, {https://github.com/HsinYingLee/MDMM}
##################################################
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import scipy.ndimage
from model_utils import conv3x3_3d, deconv3x3_3d


class NetEc(nn.Module):
    def __init__(self, opt):
        """ Content encoder network """
        super(NetEc, self).__init__()
        self.opt = opt
        enc_c = []
        tch = opt.tch
        input_dim = opt.input_dim
        enc_c += [LeakyReLUConv3d(input_dim, tch, kernel_size=3, stride=1, padding=0)]
        num_layers = 2

        for i in range(0, num_layers):
            enc_c += [ReLUINSConv3d(tch, tch*2, kernel_size=3, stride=2, padding=1)]
            tch *= 2

        for i in range(0, 3):
            enc_c += [INSResBlock(tch, tch)]

        for i in range(0, 1):
            enc_c += [INSResBlock(tch, tch)]
            enc_c += [GaussianNoiseLayer()]
        self.conv = nn.Sequential(*enc_c)

    def forward(self, x, mask=None):
        out = self.conv(x)
        return out


class NetEa(nn.Module):
    def __init__(self, opt):
        """ Attribute encoder network """
        super(NetEa, self).__init__()
        norm_layer = None
        input_dim = opt.input_dim
        c_dim = opt.num_domains
        self.opt = opt
        self.epsilon = 1e-6
        ndf = opt.tch
        n_blocks=3
        max_ndf = 4
        out_n = opt.nz
        nl_layer = get_non_linearity(layer_type='lrelu')
        conv_layers = [nn.ReplicationPad3d(1)]
        conv_layers += [nn.Conv3d(input_dim, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n+1)  # 2**n
            conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        self.conv_layers_mu = BasicBlock(output_ndf, 1, norm_layer, nl_layer)
        self.conv_layers_var = BasicBlock(output_ndf, 1, norm_layer, nl_layer)
        self.conv = nn.Sequential(*conv_layers)
        self.fc_class = nn.Linear(out_n, c_dim)
        self.fc_reg = nn.Linear(out_n, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, z=None):
        if z is None:
            x_conv = self.conv(x)
            output = self.conv_layers_mu(x_conv)
            outputVar = self.conv_layers_var(x_conv)
            output = output.view(output.size(0),-1)
            outputVar = outputVar.view(outputVar.size(0),-1)
        else:
            output = z.view(z.size(0), -1)
            outputVar = None
        output_cls = self.softmax(self.fc_class(output))
        if self.opt.regression:
            output_reg = self.fc_reg(output)
        else:
            output_reg = None
        return output, outputVar, output_cls, output_reg


class NetGen(nn.Module):
    def __init__(self, opt):
        """ Generator network """
        super(NetGen, self).__init__()
        self.nz = opt.nz
        output_dim = opt.input_dim
        self.opt = opt
        tch = opt.tch*4
        conv_z = [INSResBlock(1, opt.tch)]
        tch = tch + opt.tch
        dec1 = []
        for i in range(0, 3):
            dec1 += [INSResBlock(tch, tch)]

        dec2 = [_make_layer(self, BasicBlock_up_3d, tch//2, 1, down=False, in_planes=tch, use_upsample=False)]
        tch = tch//2
        dec3 = [_make_layer(self, BasicBlock_up_3d, tch//2, 1, down=False, in_planes=tch, use_upsample=False)]
        tch = tch//2

        # if data between -1 to 1 use Tanh, otherwise use Sigmoid
        dec4 = [nn.ConvTranspose3d(tch, output_dim, kernel_size=1, stride=1, padding=0)]+[nn.Tanh()]
        # dec4 = [nn.ConvTranspose3d(tch, output_dim, kernel_size=1, stride=1, padding=0)]+[nn.Sigmoid()]

        self.dec1 = nn.Sequential(*dec1)
        self.dec2 = nn.Sequential(*dec2)
        self.dec3 = nn.Sequential(*dec3)
        self.dec4 = nn.Sequential(*dec4)
        self.conv_z = nn.Sequential(*conv_z)
        self.upsample = nn.Upsample(scale_factor=4)

    def forward(self, x, z, c):
        z = z.view(x.size(0), 1, 8, 10, 8)
        z_up = self.upsample(z)
        z_up = self.conv_z(z_up)
        x_c_z = torch.cat([x, z_up], 1)
        out1 = self.dec1(x_c_z)
        out2 = self.dec2(out1)
        out3 = self.dec3(out2)
        out4 = self.dec4(out3)
        return out4


class NetDis(nn.Module):
    def __init__(self, opt):
        """ Discriminator network """
        super(NetDis, self).__init__()
        tch = opt.tch
        input_dim = opt.input_dim
        norm = opt.dis_norm
        sn = opt.dis_spectral_norm
        c_dim = opt.num_domains
        n_layer = 6
        self.model, curr_dim = self._make_net(tch, input_dim, n_layer, norm, sn)
        self.conv1 = nn.Conv3d(curr_dim, 1, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv3d(curr_dim, c_dim, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv3d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] #16
        tch = ch
        for i in range(1, n_layer-1):
            model += [LeakyReLUConv3d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] # 8
            tch *= 2
        model += [LeakyReLUConv3d(tch, tch, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)] # 2
        return nn.Sequential(*model), tch

    def forward(self, x):
        h = self.model(x)
        out = self.conv1(h)
        out = self.pool(out)
        out_cls = self.conv2(h)
        out_cls = self.pool(out_cls)
        return out.view(out_cls.size(0), -1), out_cls.view(out_cls.size(0), -1)

class NetDisContent(nn.Module):
    def __init__(self, opt):
        """ Content discriminator network """
        super(NetDisContent, self).__init__()
        c_dim = opt.num_domains
        tch = opt.tch*4
        model = []
        model += [LeakyReLUConv3d(tch, tch, kernel_size=3, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv3d(tch, tch, kernel_size=3, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv3d(tch, tch, kernel_size=3, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv3d(tch, tch, kernel_size=4, stride=1, padding=0)]
        self.avgpool = nn.AdaptiveAvgPool3d((1))
        self.model = nn.Sequential(*model)
        self.fc_class = nn.Linear(tch, c_dim)

    def forward(self, x, mode='cls'):
        out = self.model(x)
        out = self.avgpool(out)
        output_cls = out.view(out.size(0), -1)
        output_cls = self.fc_class(output_cls)
        return output_cls

####################################################################
#------------------------- Basic Functions -------------------------
####################################################################


def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=-1)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    #  SGDR: Stochastic Gradient Descent with Warm Restarts- CosineAnnealingWarmRestarts
    elif opts.lr_policy == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=0.0000001, last_epoch=-1) # T_0 = 6480 (dan)
    else:
        return NotImplementedError('no such learn rate policy')
    return scheduler


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool3d(kernel_size=2, stride=2)]
    sequence += [nn.Conv3d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += conv3x3(inplanes, outplanes)
    sequence += [nn.AvgPool3d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def conv3x3(in_planes, out_planes):
    return [nn.ReplicationPad3d(1), nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

####################################################################
#-------------------------- Basic Blocks --------------------------
####################################################################


# The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1, 1))
        return

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += conv3x3(inplanes, inplanes)
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class LeakyReLUConv3d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv3d, self).__init__()
        model = []
        model += [nn.ReplicationPad3d(padding)]
        if sn:
            model += [spectral_norm(nn.Conv3d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv3d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if 'norm' == 'Instance':
            model += [nn.InstanceNorm3d(n_out, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class ReLUINSConv3d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv3d, self).__init__()
        model = []
        model += [nn.ReplicationPad3d(padding)]
        model += [nn.Conv3d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        model += [nn.InstanceNorm3d(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class INSResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.InstanceNorm3d(planes)]
        model += [nn.ReLU(inplace=True)]
        model += self.conv3x3(planes, planes)
        model += [nn.InstanceNorm3d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def conv3x3(self, inplanes, out_planes, stride=1):
        return [nn.ReplicationPad3d(1), nn.Conv3d(inplanes, out_planes, kernel_size=3, stride=stride)]

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class MisINSResBlock(nn.Module):
    def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
        super(MisINSResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm3d(dim))
        self.conv2 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm3d(dim))
        self.blk1 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        self.blk2 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        model = []
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.blk1.apply(gaussian_weights_init)
        self.blk2.apply(gaussian_weights_init)

    def conv3x3(self, dim_in, dim_out, stride=1):
        return nn.Sequential(nn.ReplicationPad3d(1), nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=stride))

    def conv1x1(self, dim_in, dim_out):
        return nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3), x.size(4))
        o1 = self.conv1(x)
        o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
        o3 = self.conv2(o2)
        out = self.blk2(torch.cat([o3, z_expand], dim=1))
        out += residual
        return out


class GaussianNoiseLayer(nn.Module):
    def __init__(self,):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
        return x + noise


class ReLUINSConvTranspose3d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose3d, self).__init__()
        model = []
        model += [nn.ConvTranspose3d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
        model += [LayerNorm(n_out)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class BasicBlock_up_3d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, use_upsample=False):
        super(BasicBlock_up_3d, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = deconv3x3_3d(inplanes, planes, use_upsample=use_upsample)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3_3d(planes, planes)
        self.shortcut = deconv3x3_3d(inplanes, planes, use_upsample=use_upsample)
        self.upsample = nn.Upsample(scale_factor=2)
        self.stride = stride
        self.avpool = nn.AvgPool3d(kernel_size=2, stride=1)
        self.use_upsample = use_upsample
        self.replication_pad = nn.ReplicationPad3d(1) # same as ReflectionPad2d, but for 3d
        self.layer_norm = LayerNorm(planes)

    def forward(self, x):
        if self.use_upsample:
            out = self.upsample(x)
            out = self.replication_pad(out)
            out = self.conv1(out)
            out = self.layer_norm(out)
            out = self.relu(out)
        else:
            out = self.conv1(x)
            out = self.avpool(out)
            out = self.layer_norm(out)
            out = self.relu(out)
        return out


def _make_layer(self, block, planes, blocks, stride=1, down=True, in_planes=None, use_upsample=False):
    if in_planes:
        self.inplanes=in_planes
    downsample = None
    if down:
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv3x3_3d(self.inplanes, planes * block.expansion, stride)

    layers = []
    if not down:
        layers.append(block(self.inplanes, planes, stride, use_upsample=use_upsample))
    else:
        layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

####################################################################
#--------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################


class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u

    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))

    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))


class GaussianLayer(nn.Module):
    def __init__(self, input_dim):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReplicationPad3d(10),
            nn.Conv3d(input_dim, input_dim, 21, stride=1, padding=0, bias=None, groups=input_dim)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n = np.zeros((21, 21))
        n[10, 10] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=3)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
