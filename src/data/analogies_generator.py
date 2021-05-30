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

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import rgb2lab


class AnalogiesImagenet(Dataset):
    def __init__(self, analogies_index, root_dir, device, input_shape=224):
        self._data = np.load(analogies_index)
        self._root_dir = root_dir
        self._item_weights = [0.6, 0.3, 0.1]  # top1, top5, random
        self.device = device
        self._to_tensor = transforms.ToTensor()
        self._resize = transforms.Resize(input_shape)
        self._toPil = transforms.ToPILImage()

    def _rgb2lab(self, batch):
        return rgb2lab(batch.unsqueeze(0), self.device)[0]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = np.random.choice([2, 3, 4], p=self._item_weights)
        info = self._data[idx]
        cat, target_idx = info[:2].astype('int')
        reference_idx = int(info[item])

        target = io.imread("%s/%d/%d.JPEG" % (self._root_dir, cat, target_idx))
        reference = io.imread("%s/%d/%d.JPEG" % (self._root_dir, cat, reference_idx))
        target = self._toPil(target)
        target = self._resize(target)
        target = self._to_tensor(target)
        reference = self._toPil(reference)
        reference = self._resize(reference)
        reference = self._to_tensor(reference)

        return self._rgb2lab(target), self._rgb2lab(reference)
