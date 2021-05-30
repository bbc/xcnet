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
import csv
import os
import tarfile
import shutil
import cv2
from tqdm import tqdm

getattr(tqdm, '_instances', {}).clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-r', '--reshape', type=bool, default=True)
    parser.add_argument('-s', '--shape', type=int, default=256, help="square input shape")

    args = parser.parse_args()
    data_shape = (args.shape, args.shape)

    # correct data_path
    if args.data_path[-1] == "/":
        args.data_path = args.data_path[:-1]

    # check tar file in data_path
    assert os.path.isfile("%s/ILSVRC2012_img_train.tar" % args.data_path), \
        AssertionError("ILSVRC2012_img_train.tar not found")

    synsets_path = args.data_path + "/synsets"
    img_path = args.data_path + "/img"

    if not os.path.exists(synsets_path):
        os.mkdir(synsets_path)
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    # print("Extracting synsets:")
    os.system("pv %s/ILSVRC2012_img_train.tar | tar -x -C %s" % (args.data_path, synsets_path))

    print("Processing data:")
    with open('src/data/synset_words.txt', 'r') as f:
        labels = {row[0]: row[-1] for row in csv.reader(f, delimiter=" ")}
        for idx, label in enumerate([*labels]):
            if idx > 0: break
            print('{0}: {1}/{2}'.format(label, str(idx), str(999)))
            in_file = os.path.join(synsets_path, "%s.tar" % label)
            out_file = os.path.join(img_path, str(idx))
            if not os.path.exists(out_file):
                os.mkdir(out_file)

            names = tarfile.open(in_file).getnames()
            os.system("pv %s | tar -x -C %s" % (in_file, out_file))

            for idx_name, name in enumerate(tqdm(names, ascii=True, ncols=70)):
                sample = cv2.imread(os.path.join(out_file, name))
                if args.reshape:
                    sample = cv2.resize(sample, data_shape, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(out_file, "%d.JPEG" % idx_name), sample)
                os.remove(os.path.join(out_file, name))
