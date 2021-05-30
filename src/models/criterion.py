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
from torch import nn

from .discriminator import *


class Criterion(nn.Module):
    def __init__(self, cf, device):
        super(Criterion, self).__init__()
        self.weight_dict = self._process_dict({'histogram_loss': cf.histogram_loss_coef,
                                               'pixel_loss': cf.pixel_loss_coef,
                                               'total_variation_loss': cf.total_variance_loss_coef,
                                               'gan_loss': cf.gan_loss_coef})
        self._compute_gan_loss = cf.gan_loss_coef > 0
        self._hist_loss_d = cf.hist_loss_d
        self._hist_loss_d = cf.hist_loss_d
        self._pool_layer = torch.nn.AvgPool2d(2)
        self._hist_loss_z = torch.arange(-1, 1 + self._hist_loss_d, self._hist_loss_d).to(device)
        self._prediction_heads = 4

        if self._compute_gan_loss:
            self._gan_loss = GANLoss().to(device)
            self.discriminators, self.optimizers = [], []
            for _ in range(self._prediction_heads):
                d = discriminator(cf.disc_input_nc, cf.disc_ndf, cf.disc_norm)
                d.to(device)
                optimizer = torch.optim.Adam(d.parameters(), lr=cf.lr)
                self.discriminators.append(d)
                self.optimizers.append(optimizer)

        self._smoothL1Loss = nn.SmoothL1Loss()
        self._mseLoss = nn.MSELoss()

    @staticmethod
    def _process_dict(input_dict):
        output_dict = {}
        for (k, v) in input_dict.items():
            assert isinstance(v, (float, int))
            if v != 0.:
                output_dict[k] = v
        return output_dict

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _histogram(self, batch):
        h, w = batch.shape[2:]
        pre_a = (batch[:, 1].flatten(1) - self._hist_loss_z[:, None, None]).transpose(0, 1)
        pre_b = (batch[:, 2].flatten(1) - self._hist_loss_z[:, None, None]).transpose(0, 1)
        ha = torch.clamp(0.1 - torch.abs(pre_a), 0)
        hb = torch.clamp(0.1 - torch.abs(pre_b), 0).transpose(2, 1)
        return torch.bmm(ha, hb) / (h * w * (self._hist_loss_d ** 2))

    def _histogram_loss(self, prediction, reference):
        th = self._histogram(prediction)
        rh = self._histogram(reference)
        loss = ((th - rh) ** 2) / (th + rh + 1e-5)
        loss = 2 * torch.sum(loss.flatten(1), 1)
        return torch.mean(loss)

    def _total_variation_loss(self, prediction):
        def _tensor_size(t):
            return t.size()[1] * t.size()[2] * t.size()[3]

        batch_size = prediction.size()[0]
        h_x = prediction.size()[2]
        w_x = prediction.size()[3]
        count_h = _tensor_size(prediction[:, :, 1:, :])
        count_w = _tensor_size(prediction[:, :, :, 1:])
        h_tv = torch.pow((prediction[:, :, 1:, :] - prediction[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((prediction[:, :, :, 1:] - prediction[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _pixel_loss(self, prediction, target):
        return self._smoothL1Loss(prediction, target)

    def _discriminator_loss(self, prediction, target, head):
        disc = self.discriminators[head]
        optimiser = self.optimizers[head]
        self.set_requires_grad(disc, True)
        optimiser.zero_grad()
        loss_fake = self._gan_loss(disc(prediction.detach()), False)
        loss_real = self._gan_loss(disc(target), True)
        d_loss = (loss_fake + loss_real) * 0.5
        return d_loss

    def _generator_loss(self, prediction, head):
        disc = self.discriminators[head]
        pred_fake = disc(prediction)
        return self._gan_loss(pred_fake, True)

    def forward(self, predictions, target, reference, validate=False):
        losses = {}
        d_loss, g_loss, total_loss = 0, 0, 0
        targets, references = [target], [reference]
        for head in range(self._prediction_heads - 1):
            targets.append(self._pool_layer(targets[-1]))
            references.append(self._pool_layer(references[-1]))

        for head, target in enumerate(targets):
            predictions[head] = torch.cat((target[:, [0], :, :], predictions[head]), 1)

        # backward discriminator
        if self._compute_gan_loss and not validate:
            for head in range(self._prediction_heads):
                d_loss_head = self._discriminator_loss(predictions[head], targets[head], head)
                losses["d_loss_%d" % head] = d_loss_head
                d_loss += d_loss_head
            d_loss.backward()
            losses["d_loss"] = d_loss

            for disc, optimiser in zip(self.discriminators, self.optimizers):
                optimiser.step()
                self.set_requires_grad(disc, False)

        loss_head = 0
        for head in range(self._prediction_heads):
            for loss in self.weight_dict.keys():
                if loss == 'histogram_loss':
                    loss_head = self._histogram_loss(predictions[head], references[head])
                    losses["hist_loss_%d" % head] = loss_head
                elif loss == 'pixel_loss':
                    loss_head = self._pixel_loss(predictions[head], targets[head])
                    losses["pix_loss_%d" % head] = loss_head
                elif loss == 'total_variation_loss':
                    loss_head = self._total_variation_loss(predictions[head])
                    losses["tv_loss_%d" % head] = loss_head
                elif loss == 'gan_loss' and not validate:
                    loss_head = self._generator_loss(predictions[head], head)
                    losses["g_loss_%d" % head] = loss_head
                    g_loss += loss_head
                total_loss += self.weight_dict[loss] * loss_head

        losses["g_loss"] = g_loss
        return losses, total_loss


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
