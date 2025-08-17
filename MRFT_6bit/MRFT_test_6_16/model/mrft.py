# @Author: Xin Wen

import torch
import torch.nn as nn
from model.attention import TransformerLayerGlobal, TransformerLayerLocal, TransformerLayerDecoder
from model.position import PositionEmbeddingLearned, PositionEmbeddingSine
from einops import rearrange, repeat
from model import common
import torch.nn.functional as F

def make_model(args, parent=False):
    if args.stage == 1:
        return MRFT_global()
    else:
        return MRFT(args)



class InputFeature(nn.Module):
    def __init__(self, n_feats, n_colors, scale=16, kernel_size=3, pos_encoding_type='learned', activate=nn.PReLU):
        super().__init__()
        self.scale= scale
        self.head = nn.Sequential(
            common.default_conv(n_colors, n_feats, kernel_size),
            nn.PReLU(n_feats),
            common.ResBlock(common.default_conv, n_feats, kernel_size, act=activate(n_feats)),
            common.ResBlock(common.default_conv, n_feats, kernel_size, act=activate(n_feats))
        )
        self.projection = nn.Conv2d(n_feats * (scale * scale), n_feats, 1, 1)
        if pos_encoding_type == 'sin':
            self.position_encoding = PositionEmbeddingSine(n_feats // 2)
        else:
            self.position_encoding = PositionEmbeddingLearned(n_feats // 2)

    def forward(self, x):
        x = self.head(x)
        low_level_fea = x
        x = rearrange(x, 'b c (h m) (w n) -> b (m n c) h w', m=self.scale, n=self.scale)  # shuffling
        x = self.projection(x)
        pos_encoding = self.position_encoding(x)
        x = x + pos_encoding
        return x, low_level_fea


class Generator(nn.Module):
    def __init__(self, n_feats, n_colors=3, scale=16, kernel_size=3):
        super(Generator, self).__init__()
        self.decoder_conv = nn.Sequential(
            common.Upsampler(common.default_conv, scale, n_feats, act='prelu'),
            common.default_conv(n_feats, n_feats, kernel_size)
        )
        self.tail = nn.Sequential(
            common.default_conv(n_feats, n_feats, kernel_size),
            nn.PReLU(n_feats),
            common.default_conv(n_feats, n_colors, kernel_size)
        )

    def forward(self, x, low_level_fea):
        fea = self.decoder_conv(x)
        out = fea + low_level_fea
        out = self.tail(out)
        return out


class MRFT_global(nn.Module):
    def __init__(self, n_colors=3, scale = 16, n_feats=64, hidden_dim=256,
                 groups=1, dropout=0.1, pos_encoding_type='learned'):
        super().__init__()
        act = nn.PReLU
        kernel_size = 3
        self.input_fea = InputFeature(n_feats, n_colors, scale, kernel_size, pos_encoding_type, act)
        self.global_encoder = TransformerLayerGlobal(n_feats, n_feats, hidden_dim, groups, dropout)
        self.generator = Generator(n_feats, n_colors, scale, kernel_size)

    def forward(self, x):
        # shape of x: b*3*H*W
        identi = x
        x, low_level_fea = self.input_fea(x)
        global_fea = self.global_encoder(x)
        out = self.generator(global_fea, low_level_fea)
        return out + identi


class MRFT(nn.Module):
    def __init__(self, args, scale=16, n_colors=3, n_feats=64, hidden_dim=256, kernels=[1, 3, 5, 7, 9],
                 groups=1, dropout=0.1, pos_encoding_type='learned', kernel_size=3):
        super().__init__()
        act = nn.PReLU
        self.global_branch = MRFT_global(n_colors, scale, n_feats, hidden_dim,
                 groups, dropout, pos_encoding_type)
        self.input = InputFeature(n_feats, n_colors, scale, kernel_size, pos_encoding_type, act)
        self.local_branch = nn.ModuleList()
        for local_kernel in kernels:
            self.local_branch.append(
                TransformerLayerLocal(n_feats, n_feats, hidden_dim, local_kernel, local_kernel // 2, groups, dropout))
        self.decoder_trans = TransformerLayerDecoder(n_feats, n_feats, hidden_dim, groups, dropout)
        self.position_encoding_decoder = PositionEmbeddingLearned(n_feats // 2)
        self.generator = Generator(n_feats, n_colors, scale, kernel_size)

    def freeze_step1(self):
        for i in self.parameters():
            i.requires_grad = False
        for i in self.global_branch.parameters():
            i.requires_grad = True

    def freeze_step2(self):
        for i in self.parameters():
            i.requires_grad = True
        for i in self.global_branch.parameters():
            i.requires_grad = False

    def forward_step1(self, x):
        out = self.global_branch(x)
        return out

    def forward_step2(self, x):
        identi = x
        x, low_level_fea = self.input(x)
        global_fea = self.global_branch.global_encoder(x)
        local_feas = []
        for layer in self.local_branch:
            local_feas.append(layer(x))
        pos_encoding_decoder = self.position_encoding_decoder(x)
        for i, local_fea in enumerate(local_feas):
            if i == 0:
                fea = self.decoder_trans(global_fea + pos_encoding_decoder, local_fea)
            else:
                fea = self.decoder_trans(fea + pos_encoding_decoder, local_fea)

        out = self.generator(fea, low_level_fea)
        return out + identi

    def forward(self, x, sign=2):
        if sign == 1:
            return self.forward_step1(x)
        else:
            return self.forward_step2(x)