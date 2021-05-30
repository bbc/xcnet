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

import mkl

mkl.get_max_threads()

import argparse
import glob
import os
import pickle
import re

import faiss
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from scipy.spatial.distance import correlation as corr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import vgg19
from torchvision.transforms import Resize, Normalize, ToTensor, Compose

getattr(tqdm, '_instances', {}).clear()


class Imagenet(Dataset):
    def __init__(self, cat,
                 data_path,
                 input_shape=224):
        self.data_path = "%s/img/%d" % (data_path, cat)
        self.input_shape = input_shape
        self.filenames = self._get_filenames()
        self._normalise = Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self._toTensor = ToTensor()
        self._resize = Resize(input_shape)
        self.samples = len(self.filenames)

    @staticmethod
    def sort_list(l):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def load_img(self, name):
        img = Image.open(name)
        img = self._resize(img)
        img_y = np.array(img.convert("L"))
        img_s = self._normalise(self._toTensor(img))
        return img_y, img_s

    def _get_filenames(self):
        return self.sort_list(np.array(glob.glob("%s/*.JPEG" % self.data_path)))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.load_img(self.filenames[idx])


class RetrieveAnalogy:
    def __init__(self, retrieval_path, dataset, model, device, feature_dims=128):
        self.retrieval_path = os.path.join(retrieval_path)
        if not os.path.exists(self.retrieval_path):
            os.mkdir(self.retrieval_path)

        self.std_path = os.path.join(self.retrieval_path, 'std.p')
        self.pca_path = os.path.join(self.retrieval_path, 'pca.p')
        self.index_path = os.path.join(self.retrieval_path, 'index.f')
        self.feature_dims = feature_dims
        self._core_gen = dataset
        self.model = model
        self.device = device

    def standardize(self, features, fit=False):
        if fit:
            std = StandardScaler()
            std.fit(features)
            with open(self.std_path, "wb") as file:
                pickle.dump(std, file, pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.std_path, "rb") as file:
                std = pickle.load(file)
        return std.transform(features)

    def apply_pca(self, features, fit=False):
        if fit:
            pca = PCA(n_components=self.feature_dims)
            pca.fit(features)
            with open(self.pca_path, "wb") as file:
                pickle.dump(pca, file, pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.pca_path, "rb") as file:
                pca = pickle.load(file)
        return pca.transform(features)

    def get_index(self, features=None, fit=False):
        if fit:
            index = faiss.IndexFlatL2(self.feature_dims)
            index.add(features)
            faiss.write_index(index, self.index_path)
        else:
            index = faiss.read_index(self.index_path)
        return index

    def fit(self):
        print("Creating index:")
        gen = iter(self._core_gen)
        features = []
        for _, x in tqdm(gen, total=data.samples, ascii=True, ncols=70):
            features.append(predict(self.model, x, self.device))
        features = np.array(features)
        features = self.standardize(features, fit=True)
        features = self.apply_pca(features, fit=True)
        self.get_index(features, fit=True)

    def retrieve(self, sample, N):
        features = np.expand_dims(predict(self.model, sample, self.device), 0)
        features = self.standardize(features)
        features = self.apply_pca(features)
        index = self.get_index(features)
        _, I = index.search(features, N)
        return np.array(self._core_gen.filenames), I


class GlobalModel(nn.Module):
    def __init__(self, base):
        super(GlobalModel, self).__init__()
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier[:2]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LocalModel(nn.Module):
    def __init__(self, base):
        super(LocalModel, self).__init__()
        self.features = base.features[:30]

    def forward(self, x):
        x = self.features(x)
        return x


def predict(model, x, device):
    x = x.to(device)
    x = x.unsqueeze(0)
    with torch.no_grad():
        return model(x).cpu().numpy()[0]


def get_patch_pos(pos):
    x = (pos // 14) * 16
    y = (pos % 14) * 16
    return x, y


def compute_similarity(target_f, ref_f, target_Y, ref_Y):
    dist_csim = 0
    dist_corr = 0
    for p in range(len(target_f)):
        myvals = np.dot(ref_f, target_f[p]) / (np.linalg.norm(ref_f) * np.linalg.norm(target_f[p]))
        max_index = np.argmax(myvals)
        dist_csim += myvals[max_index]
        py, px = get_patch_pos(p)
        qy, qx = get_patch_pos(max_index)
        Cp = target_Y[py:py + 16, px:px + 16]
        Cq = ref_Y[qy:qy + 16, qx:qx + 16]
        hCp = np.histogram(np.reshape(Cp, [-1]), bins=50)[0]
        hCq = np.histogram(np.reshape(Cq, [-1]), bins=50)[0]
        dist_corr += corr(hCp, hCq)
    return dist_csim + (0.5 * dist_corr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-g', '--device', type=str, default="cuda:0")
    parser.add_argument('-N', '--N', type=int, default=5)
    parser.add_argument('-st', '--train_samples_class', type=int, default=300)
    parser.add_argument('-sv', '--val_samples_class', type=int, default=60)
    parser.add_argument('-c', '--nb_classes', type=int, default=60)

    args = parser.parse_args()

    device = torch.device(args.device)

    base = vgg19(pretrained=True, progress=False)
    model_global = GlobalModel(base).eval()
    model_local = LocalModel(base).eval()

    model_global = model_global.to(device)
    model_local = model_local.to(device)

    index_path = args.data_path + "/index"
    analogies_path = args.data_path + "/analogies"
    if not os.path.exists(index_path):
        os.mkdir(index_path)
    if not os.path.exists(analogies_path):
        os.mkdir(analogies_path)

    for mode in ["train", "val"]:
        print("Generate %s analogies:" % mode)
        mode_index_path = os.path.join(index_path, mode)
        mode_analogies_path = os.path.join(analogies_path, mode)
        if not os.path.exists(mode_index_path):
            os.mkdir(mode_index_path)
        if not os.path.exists(mode_analogies_path):
            os.mkdir(mode_analogies_path)

        nb_samples_class = args.train_samples_class if mode == "train" else args.val_samples_class
        hf = np.empty([nb_samples_class * args.nb_classes, 5]).astype("int")

        nb_samples = 0
        # for cat in range(args.nb_classes):
        for cat in range(1):
            print("Class: %d/999" % cat)
            data = Imagenet(cat, data_path=args.data_path)
            cat_path = mode_index_path + "/%d" % cat
            index = RetrieveAnalogy(cat_path, data, model_global, device)
            if mode == "train": index.fit()
            gen = iter(data)

            selection = np.arange(data.samples)
            np.random.shuffle(selection)
            selection = selection[:nb_samples_class]

            print("Creating analogies:")
            for sample_idx in tqdm(range(data.samples), ascii=True, ncols=70):
                target_y, target_s = next(gen)
                if sample_idx not in selection:
                    continue
                target_f = np.reshape(predict(model_local, target_s, device), [-1, 512])
                fn, I = index.retrieve(target_s, args.N)

                local_sim = []
                for name in fn[I][0][1:]:
                    ref_y, ref_s = data.load_img(name)
                    ref_f = np.reshape(predict(model_local, ref_s, device), [-1, 512])
                    local_sim.append(compute_similarity(target_f, ref_f, target_y, ref_y))

                sim_idx = I[0][1:][np.argmin(local_sim)]
                top5_idx = np.random.choice(I[0][1:])
                random_idx = np.random.choice(np.delete(np.arange(data.samples), sample_idx))
                hf[nb_samples] = np.array([cat, sample_idx, sim_idx, top5_idx, random_idx])
                nb_samples += 1

        np.save("%s/analogies.npy" % mode_analogies_path, hf)
