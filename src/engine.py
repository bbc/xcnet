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

import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

getattr(tqdm, '_instances', {}).clear()


class Logger:
    def __init__(self, cf, dataloader, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.checkpoint_dir = output_dir + "/checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self._writer = SummaryWriter(output_dir)

        self.display_freq = cf.display_freq
        self.checkpoint_freq = cf.checkpoint_freq

        self._meters = {}

        self._epoch = 1
        self._step = 0

        self._dataloader = dataloader
        self._nb_epochs = cf.epochs

    def update_logs(self, loss_dict, total_loss, write=False):
        loss_dict["loss"] = total_loss
        self._meters.update(loss_dict)
        out_dict = {k: "%.3f" % v for (k, v) in self._meters.items()}
        self.loop.set_postfix(loss=out_dict["loss"], g_loss=out_dict["g_loss"], d_loss=out_dict["d_loss"])
        if write:
            self.update_writer()

    def update_val_logs(self, val_meters):
        for k, v in val_meters.items():
            self._writer.add_scalar("val_%s" % k, v, self._epoch)

    def update_writer(self):
        assert self._meters.__len__() > 0
        for k, v in self._meters.items():
            assert isinstance(k, str)
            self._writer.add_scalar(k, v, self._step)
        self._step += 1

    def save_checkpoint(self, model):
        torch.save({
            'epoch': self._epoch,
            'step': self._step,
            'model_state_dict': model.state_dict()
        }, self.checkpoint_dir + "/epoch:%d-step:%d.pth" % (self._epoch, self._step))

    def on_epoch_start(self):
        self.loop = tqdm(enumerate(self._dataloader), total=len(self._dataloader), ncols=130, ascii=True)
        self.loop.set_description("Epoch [%d/%d]" % (self._epoch, self._nb_epochs))

    def on_epoch_end(self):
        self._epoch += 1


def train_one_epoch_gray(model, criterion, logger, optimizer, device):
    model.train()
    criterion.train()
    logger.on_epoch_start()
    for step, (target, reference) in logger.loop:
        target = target.to(device)
        reference = reference.to(device)

        optimizer.zero_grad()

        tgt_luma_input = target[:, [0], :, :]
        tgt_input = torch.cat((tgt_luma_input, tgt_luma_input, tgt_luma_input), dim=1)
        ref_luma_input = reference[:, [0], :, :]
        ref_luma_input = torch.cat((ref_luma_input, ref_luma_input, ref_luma_input), dim=1)
        predictions = model(tgt_input, ref_luma_input, reference)

        loss_dict, total_loss = criterion([*predictions], target, reference)

        total_loss.backward()
        optimizer.step()

        logger.update_logs(loss_dict, total_loss, step % logger.display_freq == 0 and step != 0)

        if step % logger.checkpoint_freq == 0 and step != 0:
            logger.save_checkpoint(model)

    logger.on_epoch_end()


def train_one_epoch(model, criterion, logger, optimizer, device):
    model.train()
    criterion.train()
    logger.on_epoch_start()
    for step, (target, reference) in logger.loop:
        target = target.to(device)
        reference = reference.to(device)

        optimizer.zero_grad()

        luma_input = target[:, [0], :, :]
        tgt_input = torch.cat((luma_input, luma_input, luma_input), dim=1)
        predictions = model(tgt_input, reference)

        loss_dict, total_loss = criterion([*predictions], target, reference)

        total_loss.backward()
        optimizer.step()

        logger.update_logs(loss_dict, total_loss, step % logger.display_freq == 0 and step != 0)

        if step % logger.checkpoint_freq == 0 and step != 0:
            logger.save_checkpoint(model)

    logger.on_epoch_end()


def validate(model, criterion, val_data_loader, logger, device):
    criterion.eval()
    model.eval()
    loop = tqdm(val_data_loader, total=len(val_data_loader), ncols=130, ascii=True)
    loop.set_description("[validation]")

    totals = {}
    step = 0
    with torch.no_grad():
        for target, reference in loop:
            target = target.to(device)
            reference = reference.to(device)

            luma_input = target[:, [0], :, :]
            tgt_input = torch.cat((luma_input, luma_input, luma_input), dim=1)

            prediction = model(tgt_input, reference)
            colour_prediction = torch.cat((luma_input, prediction), 1)
            loss_dict, _ = criterion(colour_prediction, target, reference)
            for (k, v) in loss_dict.items():
                item = v.item()
                totals[k] += item if k in totals else item
            step += 1

    for (k, v) in totals.items():
        totals[k] /= len(val_data_loader)
    logger.update_val_logs(totals)
