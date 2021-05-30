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

from torch import nn
from torch.hub import load_state_dict_from_url

from .layers import AttentionBlock, ProjectionBlock, DecoderBlock, HeadBlock, FeaturesExtractor

__all__ = ["xcnet", "xcnet_no_pos"]

model_urls = {
    "xcnet": "https://www.dropbox.com/s/ouy9ginorn2ccuw/xcnet_model.pth?dl=1",
    "xcnet_no_pos": "https://www.dropbox.com/s/tfszakdyrsfuvsm/xcnet_no_pos_model.pth?dl=1"
}


class __XCNET__(nn.Module):
    def __init__(self, input_shape=224, axial=True, scales=2, dim=256, output_nc=2, pos=True):
        super(__XCNET__, self).__init__()
        self.features = FeaturesExtractor()
        self.features.eval()

        self.proj1 = ProjectionBlock(64, dim, get_ref=False)
        self.proj2 = ProjectionBlock(128, dim, get_ref=False)
        self.proj3 = ProjectionBlock(256, dim)
        self.proj4 = ProjectionBlock(512, dim)
        self.proj5 = ProjectionBlock(512, dim)
        self.proj6 = ProjectionBlock(512, dim, get_ref=False)

        kernel = [input_shape // (2 ** k) if axial else None for k in range(2, 5)]
        self.att3 = AttentionBlock(axial, scales, dim, kernel[0], pos)
        self.att4 = AttentionBlock(axial, scales, dim, kernel[1], pos)
        self.att5 = AttentionBlock(axial, scales, dim, kernel[2], pos)

        self.dec4 = DecoderBlock(dim)
        self.dec3 = DecoderBlock(dim)
        self.dec2 = DecoderBlock(dim)
        self.dec1 = DecoderBlock(dim)

        self.head4 = HeadBlock(dim, output_nc)
        self.head3 = HeadBlock(dim, output_nc)
        self.head2 = HeadBlock(dim, output_nc)
        self.head1 = HeadBlock(dim, output_nc)

    def forward(self, tgt, ref):
        tgt1, tgt2, tgt3, tgt4, tgt5, tgt6 = self.features(tgt)
        _, _, ref3, ref4, ref5, _ = self.features(ref)

        tgt1 = self.proj1(tgt1)
        tgt2 = self.proj2(tgt2)
        tgt3, ref3 = self.proj3(tgt3, ref3)
        tgt4, ref4 = self.proj4(tgt4, ref4)
        tgt5, ref5 = self.proj5(tgt5, ref5)
        tgt6 = self.proj6(tgt6)

        fused5 = self.att5(tgt5, ref5)
        fused4 = self.att4(tgt4, ref4)
        fused3 = self.att3(tgt3, ref3)

        dec4 = self.dec4(tgt6, fused5, tgt4)
        dec3 = self.dec3(dec4, fused4, tgt3)
        dec2 = self.dec2(dec3, fused3, tgt2)
        dec1 = self.dec1(dec2, None, tgt1)

        pred4 = self.head4(dec4)
        pred3 = self.head3(dec3)
        pred2 = self.head2(dec2)
        pred1 = self.head1(dec1)

        return pred1, pred2, pred3, pred4


def xcnet(input_shape=224, axial=True, scales=2, dim=256, output_nc=2, pos=True, pretrained=False, progress=True):
    if pretrained:
        assert input_shape == 224 and axial == True and scales == 2 \
               and dim == 256 and output_nc == 2 and pos == True
    model = __XCNET__(input_shape=input_shape, axial=axial, scales=scales, dim=dim, output_nc=output_nc, pos=pos)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["xcnet"], map_location='cpu', progress=progress)
        model.load_state_dict(state_dict["model_state_dict"])
    return model


def xcnet_no_pos(input_shape=224, axial=True, scales=2, dim=256, output_nc=2, pos=False, pretrained=False,
                 progress=True):
    if pretrained:
        assert input_shape == 224 and axial == True and scales == 2 \
               and dim == 256 and output_nc == 2 and pos == False
    model = __XCNET__(input_shape=input_shape, axial=axial, scales=scales, dim=dim, output_nc=output_nc, pos=pos)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["xcnet_no_pos"], map_location='cpu', progress=progress)
        model.load_state_dict(state_dict["model_state_dict"])
    return model
