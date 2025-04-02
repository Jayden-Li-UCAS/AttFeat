import math
import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer

from turtle import forward
from numpy import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as KF
import torchvision.models as models
from lib.modules import *

import torch.nn as nn
import torch
import math
from typing import Tuple


try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


def window_partition(x, window_size: Tuple[int, int]):
    B, H, W, C = x.shape
    _assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    _assert(W % window_size[1] == 0, f'width ({W}) must be divisible by window ({window_size[1]})')
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x

class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, channel_last, bias: bool = True):
        super().__init__()

        proj_out_dim = dim_out * 2
        self.proj = nn.Linear(dim_in, proj_out_dim, bias=bias) if channel_last else \
            nn.Conv2d(dim_in, proj_out_dim, kernel_size=1, stride=1, bias=bias)

        self.channel_dim = -1 if channel_last else 1

        self.act_layer = nn.GELU()

    def forward(self, x: torch.Tensor):
        x, gate = torch.tensor_split(self.proj(x), 2, dim=self.channel_dim)
        return x * self.act_layer(gate)


class MLP(nn.Module):
    def __init__(self, dim, channel_last, expansion_ratio, gated=True, bias=True, drop_prob: float = 0.):
        super().__init__()

        inner_dim = int(dim * expansion_ratio)
        if gated:
            inner_dim = math.floor(inner_dim * 2 / 3 / 32) * 32
            proj_in = GLU(dim_in=dim, dim_out=inner_dim, channel_last=channel_last, bias=bias)
        else:
            proj_in = nn.Sequential(
                nn.Linear(in_features=dim, out_features=inner_dim, bias=bias),
                nn.GELU(),
            )
        self.net = nn.Sequential(
            proj_in,
            nn.Dropout(p=drop_prob),
            nn.Linear(in_features=inner_dim, out_features=dim, bias=bias)
        )

    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float=1e-5, inplace: bool=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma

class SelfAttention(nn.Module):
    def __init__(self, dim, dim_head=32, bias=True):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.window_size=(8,8)
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor,q: torch.Tensor,k: torch.Tensor,v: torch.Tensor):
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        q = q.view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)
        v= v.view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        ###################################################################
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        return x


class W_Attention(nn.Module):
    def __init__(self, dim, partition_type, skip_first_norm, ls_init_value=1e-5, dim_head=32, drop_path=0.0,
                 partition_size=(8, 8)):
        super().__init__()

        self.partition_type = partition_type
        self.partition_size = partition_size

        partition_size = tuple(partition_size)
        assert len(partition_size) == 2

        self.norm1 = nn.Identity() if skip_first_norm else nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, dim_head=dim_head, bias=True)
        self.ls1 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.ls2 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        self.mlp = MLP(dim=dim,
                       channel_last=True,
                       expansion_ratio=4,
                       gated=False,
                       bias=True,
                       drop_prob=0.0)

    def _partition_attn(self, x,q,k,v):
        img_size = x.shape[1:3]

        H, W = img_size
        H_win, W_win = self.partition_size
        pad_l = pad_t = 0
        pad_r = (W_win - W % W_win) % W_win
        pad_b = (H_win - H % H_win) % H_win
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        partitioned = window_partition(x, self.partition_size)
        partitioned = self.self_attn(partitioned,q,k,v)
        x = window_reverse(partitioned, self.partition_size, (Hp, Wp))
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x

    def forward(self, x,q,k,v):
        x = self._partition_attn(x,q,k,v)
        return x


class Attention_Block(nn.Module):
    def __init__(self, dim, skip_first_norm):
        super().__init__()

        self.att_window = W_Attention(dim=dim, partition_type='WINDOW', skip_first_norm=False)

    def forward(self, x,q,k,v):
        x = self.att_window(x,q,k,v)
        return x

#############################################################################################################

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, bias=False,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # self.bias = bias
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=bias))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]))

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)

        return x


class SA_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=1,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = 128
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()
        ##############

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # sueeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)

        xx = self.sigmoid(xx)
        return xx,q,k,v


class Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.SA_attn = SA_Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                                  activation=act_layer, norm_cfg=norm_cfg)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)
        self.W_attn = Attention_Block(dim=128, skip_first_norm=False)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x0):
        x0_norm = x0.permute(0, 2, 3, 1)
        x0_norm=self.norm1(x0_norm)
        x0_norm = x0_norm.permute(0, 3, 1, 2)
        x1,q,k,v = self.SA_attn(x0_norm)

        x2=x0_norm.permute(0,2,3,1)
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        x2=self.W_attn(x2,q,k,v)
        x2 = x2.permute(0, 3, 1, 2)

        x_att= x1 * x2
        x_att = x0 + self.drop_path(x_att)
        x_att_norm = x_att.permute(0, 2, 3, 1)
        x_att_norm=self.norm2(x_att_norm)
        x_att_norm = x_att_norm.permute(0, 3, 1, 2)
        x_att_out = x_att + self.drop_path(self.mlp(x_att_norm))
        return x_att_out


class BasicLayer(nn.Module):
    def __init__(self,  embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=nn.ReLU6):
        super().__init__()

        self.transformer_blocks = nn.ModuleList()
        self.transformer_blocks.append(Block(
            embedding_dim, key_dim=key_dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
            drop=drop, drop_path=drop_path if isinstance(drop_path, list) else drop_path,
            norm_cfg=norm_cfg,
            act_layer=act_layer))

    def forward(self, x):
        # token * N
        x = self.transformer_blocks[0](x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

        self.conv3x3_2 = nn.Conv2d(channels, channels , kernel_size=3, stride=1, padding=1)
        self.bn=nn.BatchNorm2d(channels,affine=False)
        self.cgp = nn.AdaptiveAvgPool2d((None,1))

    def forward(self, x):
        b, c, h, w = x.size()                                                       ##2,32,192,192
        group_x = x.reshape(b * self.groups, -1, h, w)   # b*g,c//g,h,w              ##16(2*8),4(32/8),192,192
        x_h = self.pool_h(group_x)                                                  ##16,4,192,1
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)                              ##16,4,192,1
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))                             ##16,4,384,1
        x_h, x_w = torch.split(hw, [h, w], dim=2) ##从第二股维度切，切成h和w两份        ##16,4,192,1,   16,4,192,1
        x0 = group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()           ##16,4,192,192
        x1 = self.gn(x0)
        x2 = self.conv3x3_1(group_x)                                                ##16,4,192,192
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))    ##16,1,4
        x12 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw             ##16,4,36864
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  ##16,1,4
        x22 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw             ##16,4,36864(192*192)
        weights1 = (torch.matmul(x11, x22) + torch.matmul(x21, x12)).reshape(b * self.groups, 1, h, w)    ####16,1,192,192

        x3 = self.bn(x0.reshape(b, c, h, w))
        x4 = self.conv3x3_2(x)
        x31 = self.softmax(self.cgp(x3.permute(0, 2, 3,1)).permute(0, 3, 1,2).reshape(b , -1, h*w))
        x32 = x3.reshape(b,c,h*w).permute(0, 2,1)
        x41 = self.softmax(self.cgp(x4.permute(0, 2, 3,1)).permute(0, 3, 1,2).reshape(b , -1, h*w))
        x42 = x4.reshape(b,c,h*w).permute(0, 2,1)
        weights2 = (torch.matmul(x31, x42) + torch.matmul(x41, x32)).reshape(b , c, -1, 1)
        return (group_x * weights1.sigmoid()).reshape(b, c, h, w)+(x * weights2.sigmoid())

class BaseNet(nn.Module):
    """ Helper class to construct a fully-convolutional network that
        extract a l2-normalized patch descriptor.
    """
    ##输入通道数inchan   ##否使用空洞卷积dilated  ##空洞率dilation
    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        super(BaseNet,self).__init__()
        self.inchan = inchan
        self.curchan = inchan
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])
    ##outd: 输出通道数   ##k: 卷积核大小    ##stride: 步幅
    ##bn: 是否使用BatchNorm    ##relu: 是否使用ReLU
    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine, momentum=0.1)
    ##Batch Normlization使一批batch的特征图满足均值为0，方差为1

    def MakeBlk(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True,):  ##outd=32
        d = self.dilation * dilation  ##计算卷积层的空洞率,外部设置的乘内部
        ##conv_params配置卷积层参数,padding: 根据空洞卷积率自动计算padding大小.padding指的是在进行卷积操作时,对输入进行边缘填充
        ##dilation: 使用上一步设置的d作为空洞率  ##stride: 步幅 ##其他卷积参数使用默认值
        conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=1)
        t = nn.ModuleList([])  ##初始化nn.ModuleList()组件t
        ##创建卷积层,使用conv_params作为配置
        t.append( nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params) )
        if bn and self.bn: t.append( self._make_bn(outd) )  ##如果允许,创建BatchNorm层,使用内部方法自动配置
        if relu: t.append( nn.ReLU(inplace=True) )  ##如果允许,附加ReLU激活层
        #nn.ModuleList()本质上是一个列表,可以向其中append不同的模块。
        ##但并不是一个可调用执行的模块,我们需要转换为nn.Sequential()。
        ##将之前append进ModuleList中的层组合起来,转换成一个Sequential模块。
        blk = nn.Sequential(*t)
        self.curchan = outd
        self.dilation *= stride
        return blk

##Adapter: Adapter通常用来进行迁移学习或者模型压缩。
# 它是一个很小的神经网络模块,可以插入预训练好的Transformer模型中,通过微调这个小模块来快速适配下游任务。
# 使用Adapter可以避免重新训练整个Transformer模型,这样可以大大节省计算资源和训练时间。
class Adapter(BaseNet):
    def __init__(self, mchan=4, **kw ):
        super(Adapter,self).__init__()
        t = BaseNet()
        tt = nn.ModuleList([])
        ops = nn.ModuleList([])                   ##初始 1，3，512，512
        ops.append(t.MakeBlk(8*mchan))            ##1，32，512，512
        self.EMA_1=EMA(channels=8*mchan)
        ops.append(t.MakeBlk(16*mchan,stride=2))  ##1，64，512，512
        self.EMA_2 = EMA(channels=16*mchan)
        ops.append(t.MakeBlk(32*mchan,stride=2))  ##1，128，512，512
        self.EMA_3 = EMA(channels=32*mchan)
        self.ops = ops
        self.RLNs = tt
    def forward(self,x):  ##前向传播过程中,交替使用残差连接和标准卷积
        x = self.ops[0](x)
        x = self.EMA_1(x)+x
        x = self.ops[1](x)
        x = self.EMA_2(x)+x
        x = self.ops[2](x)
        x = self.EMA_3(x)+x
        return x


class Encoder(nn.Module):
    ##dim: 编码输出的维度,默认为128  ##mchan: 通道数的倍数,默认为4  ##空洞率dilation
    def __init__(self, dim=128, mchan=4, relu22=False, dilation=4, **kw):
        ##调用父类nn.Module的初始化方法
        super(Encoder, self).__init__()
        ##创建BaseNet对象
        t = BaseNet(inchan=32 * mchan, dilation=dilation)
        ##创建一个空的nn.ModuleList
        ops = nn.ModuleList([])
        ##向ops中依次添加3个卷积块,这些卷积块来自BaseNet的MakeBlk方法
        ops.append(t.MakeBlk(32 * mchan, k=3, stride=2, relu=False))  ##1，128，512，512
        self.EMA_1 = EMA(channels=32 * mchan)
        ops.append(t.MakeBlk(dim, k=3, stride=2, bn=False, relu=False))  ##1，128，512，512
        ##定义输出维度为传入的dim参数
        self.out_dim = dim
        ##将卷积块OPS列表赋值成类属性
        self.ops = ops

    def forward(self, x):
        x = self.ops[0](x)
        x=self.EMA_1(x) + x
        x = self.ops[1](x)
        return x


class ConditionalEstimator(nn.Module):  ##这是一个ConditionalEstimator类,用于估计条件概率。
    def __init__(self) -> None:
        super(ConditionalEstimator,self).__init__()
        self.dropout = nn.Dropout(0.1)  ##定义一个dropout层,丢弃概率为0.1
        ##三个不同大小的空洞卷积，输出通道由初始128-64-32-16，图片分辨率一直为512，512
        self.attn = BasicLayer(embedding_dim=128, key_dim=16, num_heads=8)
        self.preconv = nn.Sequential(nn.Conv2d(128,64,kernel_size=3,padding=1,bias=False),
                                    nn.BatchNorm2d(64,affine=False),
                                    nn.ReLU(),
                                    nn.Conv2d(64,32,kernel_size=3,dilation=2,padding=2,bias=False),
                                    nn.BatchNorm2d(32,affine=False),
                                    nn.ReLU(),
                                    nn.Conv2d(32,16,kernel_size=3,dilation=4,padding=4,bias=False),
                                    nn.BatchNorm2d(16,affine=False),
                                    nn.ReLU()
                                    )
        ##批标准化(BatchNorm)和实例标准化(InstanceNorm)
        self.bn1 = nn.Sequential(nn.BatchNorm2d(16,affine=False),##一定程度上抑制内部分布变化。##这里affine=False表示不使用可学习的缩放和偏移参数。
                                nn.ReLU(),## relu激活函数,引入非线性
                                nn.InstanceNorm2d(16,affine=False),##对每个样本进行特征归一化,抑制样本间分布差异。
                                nn.ReLU())
        self.bn2 = nn.Sequential(nn.BatchNorm2d(16,affine=False),
                                nn.ReLU(),
                                nn.InstanceNorm2d(16,affine=False),
                                nn.ReLU())
        self.bn3 = nn.Sequential(nn.BatchNorm2d(16,affine=False),
                                nn.ReLU(),
                                nn.InstanceNorm2d(16,affine=False),
                                nn.ReLU())
        self.pool1 = nn.AvgPool2d(3,stride=1,padding=1)##平均池化，池化核的大小是3x3，步长为1，使用1像素的零填充
        self.pool2 = nn.AvgPool2d(3,stride=1,padding=1)##所以池化完不会影响图片大小
        self.pool3 = nn.AvgPool2d(3,stride=1,padding=1)
        self.layer1 = nn.Sequential(nn.Conv2d(16,16,3,padding=1,bias=False),  ##下面定义为conv，标准的三个结构，16-16
                                    nn.BatchNorm2d(16,affine=False),
                                    nn.ReLU(),
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(16,16,3,padding=1,bias=False),
                                    nn.BatchNorm2d(16,affine=False),
                                    nn.ReLU(),
                                    )
        self.layer3 = nn.Sequential(nn.Conv2d(16,16,3,padding=1,bias=False),
                            nn.BatchNorm2d(16,affine=False),
                            nn.ReLU())
        self.postconv = nn.Sequential(nn.Conv2d(16,2,3,padding=1,bias=False))  ##将特征维度由16卷积为2

    ##LN函数的设计实际上模仿了条件密度估计的过程,通过特征提取、映射、采样、归一化等步骤去估计给定条件下的目标分布,并使估计更加可靠。
    ##条件密度估计(Conditional Density Estimation)指的是给定一些条件,估计条件下某目标随机变量的概率密度分布。这里“条件”通常指的是特征或者上下文环境。
    ##在这个模型中,条件是输入图片的特征映射x,目标随机变量就是我们想要预测的每一个pixel的值。
    ##模型模仿最大似然估计的过程,得到了给定条件x下,目标pixel分布的估计。
    def LN(self,x,conv,pool,bn):        ##输入 1，16，512，512
        x = conv(x)                     ## 1，16，512，512
        ##对x取指数函数,再进行截断,这是为了映射到指数分布并防止过大
        x = x.exp().clamp(max=1e4)      ## 1，16，512，512
        ##这里实现了一种“池化归一化”操作
        x = x/(pool(x)+1e-5)            ## 1，16，512，512
        x = bn(x)                       ## 1，16，512，512
        return x
    def forward(self,x):                                ##输入 1，128，512，512
        x = self.dropout(x)                             ## 1，128，512，512
        x = self.attn(x)
        x = self.preconv(x)                             ## 1，16，512，512
        x = self.LN(x,self.layer1,self.pool1,self.bn1)  ## 1，16，512，512
        x = self.LN(x,self.layer2,self.pool2,self.bn2)  ## 1，16，512，512
        x = self.LN(x,self.layer3,self.pool3,self.bn3)  ## 1，16，512，512
        x = self.postconv(x)                            ## 1，2，512，512
        x = F.softmax(x,dim=1)[:,0].unsqueeze(1)        ## 1，1，512，512
        return x

class Superhead(nn.Module):
    def __init__(self) -> None:
        super(Superhead,self).__init__()
        ##定义先验Density Estimator,使用1x1卷积直接预测先验概率，将维度直接从128变为1维
        self.PriorEstimator = nn.Sequential(nn.Conv2d(128,1,kernel_size=1))
        ##定义条件Density Estimator,使用Pre定义的ConditionalEstimator,得到的是1维的
        self.ConditionalEstimator = ConditionalEstimator()
    ##取值范围在[0,1]之间的概率输出
    def p_x(self,x):                 ##输入1，128，512，512
        x = self.PriorEstimator(x)   ##1，1，512，512
        p_x = F.softplus(x)          ##1，1，512，512
        p_x = p_x/(1+p_x)            ##1，1，512，512
        #p_x = x/(1+x)
        return p_x

    def forward(self,x):
        p_c = self.ConditionalEstimator(x)
        p_x = self.p_x(x)
        # p_y = F.softplus(x1+x2)
        # p_y = p_y/(1+p_y)
        # p_x = self.p_x(x)
        p_y = (p_x*p_c)    ##512*512热度图
        return p_y


class MMNet(nn.Module):
    def __init__(self):
        super(MMNet, self).__init__()
        self.ada1 = Adapter()
        self.ada2 = Adapter()
        self.enc = Encoder()     ##得到特征向量
        self.det = Superhead()   ##通过特征向量计算得到特征点，使得特征点和特征向量可以一块训练

    def forward1(self,imgs):
        feat_in = self.ada1(imgs)
        feat = self.enc(feat_in)
        score = self.det(feat.pow(2))
        return F.normalize(feat,dim=1), score

    def forward2(self,imgs):
        feat_in = self.ada2(imgs)
        feat = self.enc(feat_in)
        score = self.det(feat.pow(2))
        return F.normalize(feat,dim=1), score

    def forward(self, img1, img2):
        feat1,score1 = self.forward1(img1)
        feat2,score2 = self.forward2(img2)
        return {
            'feat': [feat1, feat2],
            'score': [score1, score2]
        }