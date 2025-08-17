# @Author: Xin Wen

from regex import D
from requests import head
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange, repeat
from torch import einsum
from einops.layers.torch import Rearrange


class AttentionLocal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

class AttentionGlobal(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        assert dim % heads == 0
        dim_head = dim // heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class AttentionGlobalCross(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        assert dim % heads == 0
        dim_head = dim // heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.to_q = nn.Linear(dim, dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        kv = self.to_kv(y).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class MLPLocal(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, output_dim, 1, 1),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = self.mlp(x)
        return x

class MLPGlobal(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.mlp(x)
        return x

class TransformerLayerLocal(nn.Module):
    def __init__(self, inchannel, outchannel, hiddenchannel, kernelsize, padding, groups=4, dropout=0):
        super().__init__()
        self.attention = AttentionLocal(inchannel, inchannel, kernelsize, padding=padding, groups=groups)
        self.norm1 = nn.GroupNorm(1, inchannel)
        self.mlp = MLPLocal(inchannel, outchannel, hiddenchannel, dropout)
        self.norm2 = nn.GroupNorm(1, outchannel)
    
    def forward(self, x):
        x = self.attention(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x

class TransformerLayerGlobal(nn.Module):
    def __init__(self, inchannel, outchannel, hiddenchannel, groups=4, dropout=0):
        super().__init__()
        self.attention = AttentionGlobal(inchannel, groups, dropout)
        self.norm1 = nn.LayerNorm(inchannel)
        self.mlp = MLPGlobal(inchannel, outchannel, hiddenchannel, dropout)
        self.norm2 = nn.LayerNorm(outchannel)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = self.attention(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x

class TransformerLayerDecoder(nn.Module):
    def __init__(self, inchannel, outchannel, hiddenchannel, groups, dropout):
        super().__init__()
        self.attention1 = AttentionGlobal(inchannel, groups, dropout)
        self.norm1 = nn.LayerNorm(inchannel)
        self.norm2 = nn.LayerNorm(outchannel)

        self.attention2 = AttentionGlobalCross(inchannel, groups, dropout)
        self.norm3 = nn.LayerNorm(inchannel)
        self.mlp2 = MLPGlobal(inchannel, outchannel, hiddenchannel, dropout)
        self.norm4 = nn.LayerNorm(outchannel)


    def forward(self, x, hidden):
        b, c, h, w = x.shape
        # self attention
        x = x.view(b, c, -1).permute(0, 2, 1)
        hidden = hidden.view(b, c, -1).permute(0, 2, 1)
        x = self.attention1(self.norm1(x)) + x
        x = self.attention2(self.norm2(x), self.norm3(hidden)) + x
        x = self.mlp2(self.norm4(x)) + x
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x