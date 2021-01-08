# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import numpy as np

import torch
import MinkowskiEngine as ME
from scipy.sparse import random
from scipy import stats


class CustomRandomState(np.random.RandomState):
    def randint(self, k):
        i = np.random.randint(k)
        return i - i % 2

np.random.seed(12345)
rs = CustomRandomState()
rvs = stats.poisson(25, loc=10).rvs


def get_coords(data):
    coords = []
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if col != ' ':
                coords.append([i, j])
    return np.array(coords)


def get_coords_lambda(data):
    coords = []
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if col != 0:
                coords.append([i, j])
    return np.array(coords)


def data_loader(nchannel=3,
                max_label=5,
                is_classification=True,
                seed=-1,
                batch_size=2):
    if seed >= 0:
        torch.manual_seed(seed)

    data = [
        "   X   ",  #
        "  X X  ",  #
        " XXXXX "
    ]

    # Generate coordinates
    coords = [get_coords(data) for i in range(batch_size)]
    coords = ME.utils.batched_coordinates(coords)

    # features and labels
    N = len(coords)
    feats = torch.randn(N, nchannel)
    label = (torch.rand(2 if is_classification else N) * max_label).long()
    return coords, feats, label


def data_loader_lambda(nchannel=30,
                max_label=5,
                is_classification=True,
                seed=-1,
                batch_size=2):
    if seed >= 0:
        torch.manual_seed(seed)

    S = random(256, 256, density=0.01, random_state=rs, data_rvs=rvs)
    data = S.A

    # Generate coordinates
    coords = [get_coords_lambda(data) for i in range(batch_size)]
    coords = ME.utils.batched_coordinates(coords)

    # features and labels
    N = len(coords)
    feats = torch.randn(N, nchannel)
    label = (torch.rand(2 if is_classification else N) * max_label).long()

    # print(coords.shape)
    # print(feats.shape)
    # print(label.shape)
    return coords, feats, label
