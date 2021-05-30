# XCNET - Attention-based Stylisation for Exemplar Image Colourisation
# Copyright (C) 2021  BBC Research & Development
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import MultiheadAttention
from torchvision.models import vgg19

from .axial_attention import MultiheadAxialAttention as axial_pos
from .axial_attention_no_pos import MultiheadAxialAttention as axial_no_pos

__all__ = ["AttentionBlock",
           "ProjectionBlock",
           "DecoderBlock",
           "HeadBlock",
           "FeaturesExtractor"]


class AttentionModule(nn.Module):
    def __init__(self, dim=256):
        super(AttentionModule, self).__init__()
        self.att = MultiheadAttention(dim, num_heads=8)
        self.bn_tgt = nn.BatchNorm2d(dim)
        self.bn_ref = nn.BatchNorm2d(dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        tgt, ref = inputs
        tgt_norm = self.bn_tgt(tgt)
        ref_norm = self.bn_ref(ref)
        bs, c, h, w = tgt_norm.shape
        tgt_feat = tgt_norm.flatten(2).permute(2, 0, 1)
        ref_feat = ref_norm.flatten(2).permute(2, 0, 1)
        fused = self.att(tgt_feat, ref_feat, ref_feat)[0]
        fused = fused.permute(1, 2, 0).view(bs, c, h, w)
        fused += tgt
        return self.activation(fused), ref


class AxialAttentionModule(nn.Module):
    def __init__(self, kernel, dim=256, pos=True):
        super(AxialAttentionModule, self).__init__()
        MultiheadAxialAttention = axial_pos if pos else axial_no_pos
        self.rows = MultiheadAxialAttention(dim, dim, kernel)
        self.cols = MultiheadAxialAttention(dim, dim, kernel, width=True)
        self.bn_tgt = nn.BatchNorm2d(256)
        self.bn_ref = nn.BatchNorm2d(256)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        tgt, ref = inputs
        tgt_norm = self.bn_tgt(tgt)
        ref_norm = self.bn_ref(ref)
        fused = self.rows(tgt_norm, ref_norm, ref_norm, need_weights=True)[0]
        fused = self.cols(fused, ref, ref, need_weights=True)[0]
        fused += tgt
        return self.activation(fused), ref


class AttentionBlock(nn.Module):
    def __init__(self, axial=True, scales=2, dim=256, kernel=None, pos=True):
        super(AttentionBlock, self).__init__()
        if axial:
            assert kernel is not None
        sequence = [AxialAttentionModule(kernel, dim, pos) if axial else AttentionModule(dim)] * scales
        self.module = nn.Sequential(*sequence)

    def forward(self, tgt, ref):
        return self.module((tgt, ref))[0]


class ProjectionBlock(nn.Module):
    def __init__(self, in_dim, dim=256, get_ref=True):
        super(ProjectionBlock, self).__init__()
        self.get_ref = get_ref
        self.proj_tgt = nn.Conv2d(in_dim, dim, kernel_size=1, padding=0)
        if get_ref:
            self.proj_ref = nn.Conv2d(in_dim, dim, kernel_size=1, padding=0)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt, ref=None):
        tgt = self.proj_tgt(tgt)
        if self.get_ref:
            ref = self.proj_ref(ref)
            return self.activation(tgt), self.activation(ref)
        else:
            return self.activation(tgt)


class DecoderBlock(nn.Module):
    def __init__(self, dim=256):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(dim + dim, dim, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def _upsample(self, x1, x2):
        x1 = self.upsample(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, src, fused, skip):
        if fused is not None:
            src = src + fused
        x = self.conv1(src)
        x = self.activation(x)
        x = self._upsample(x, skip)
        x = self.conv2(x)
        x = self.activation(x)
        return x


class HeadBlock(nn.Module):
    def __init__(self, dim=256, output_nc=2):
        super(HeadBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, output_nc, kernel_size=1, padding=0)
        self.activation = nn.ReLU(inplace=True)
        self.out_activation = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.out_activation(x)
        return x


class FeaturesExtractor(nn.Module):
    def __init__(self):
        super(FeaturesExtractor, self).__init__()
        features = vgg19(pretrained=True, progress=False).features
        self.to_relu_1_1 = nn.Sequential()
        self.to_relu_2_1 = nn.Sequential()
        self.to_relu_3_1 = nn.Sequential()
        self.to_relu_4_1 = nn.Sequential()
        self.to_relu_5_1 = nn.Sequential()
        self.to_relu_5_4 = nn.Sequential()

        for x in range(2):
            self.to_relu_1_1.add_module(str(x), features[x])
        for x in range(2, 7):
            self.to_relu_2_1.add_module(str(x), features[x])
        for x in range(7, 12):
            self.to_relu_3_1.add_module(str(x), features[x])
        for x in range(12, 21):
            self.to_relu_4_1.add_module(str(x), features[x])
        for x in range(21, 30):
            self.to_relu_5_1.add_module(str(x), features[x])
        for x in range(30, 36):
            self.to_relu_5_4.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_1(x)
        h_relu_1_1 = h
        h = self.to_relu_2_1(h)
        h_relu_2_1 = h
        h = self.to_relu_3_1(h)
        h_relu_3_1 = h
        h = self.to_relu_4_1(h)
        h_relu_4_1 = h
        h = self.to_relu_5_1(h)
        h_relu_5_1 = h
        h = self.to_relu_5_4(h)
        h_relu_5_4 = h
        out = (h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_5_1, h_relu_5_4)
        return out
