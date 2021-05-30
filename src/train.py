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

import argparse
import datetime
import random
import time
from importlib.machinery import SourceFileLoader

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.analogies_generator import AnalogiesImagenet
from engine import Logger, train_one_epoch, validate
from models.criterion import Criterion
from models.xcnet import xcnet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True, help='data path')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='output path')
    parser.add_argument('-c', '--config', type=str, required=True, help='config file')
    parser.add_argument('-g', '--device', type=str, default="cuda:0", help='gpu device')
    args = parser.parse_args()

    cf = SourceFileLoader('config', args.config).load_module()
    device = torch.device(args.device)

    torch.manual_seed(cf.seed)
    np.random.seed(cf.seed)
    random.seed(cf.seed)

    model = xcnet(input_shape=cf.input_shape, axial=cf.axial_attention, scales=cf.nb_scales,
                  dim=cf.hidden_dim, output_nc=cf.output_nc, pos=cf.pos_encoder)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    criterion = Criterion(cf, device)
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.lr)

    train_dataset = AnalogiesImagenet(args.data_path + "/analogies/train/analogies.npy",
                                      root_dir=args.data_path + "/img",
                                      device=device,
                                      input_shape=cf.input_shape)

    val_dataset = AnalogiesImagenet(args.data_path + "/analogies/val/analogies.npy",
                                    root_dir=args.data_path + "/img",
                                    device=device,
                                    input_shape=cf.input_shape)

    train_data_loader = DataLoader(train_dataset, batch_size=cf.batch_size,
                                   shuffle=True, num_workers=cf.num_workers)

    val_data_loader = DataLoader(val_dataset, batch_size=cf.batch_size,
                                 shuffle=False, num_workers=cf.num_workers)

    train_logger = Logger(cf, train_data_loader, args.output_path)
    start_time = time.time()
    for epoch in range(1, cf.epochs + 1):
        train_one_epoch(model, criterion, train_logger, optimizer, device)
        validate(model, criterion, val_data_loader, train_logger, device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
