# fcn_test

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon.data.vision import MNIST
from mxnet.gluon import nn


block1 = nn.Block()
net = nn.Sequential()

with net.name_scope():
    net.add(nn.Dense(64, activation='relu'))

symbol_data = mx.sym.var('data')
tmp = net(symbol_data)
tmp(block1)

digraph = mx.viz.plot_network(tmp)
digraph.render()


# for key, layer in model._children.items():
#     # print(key, layer)
#     print(layer.collect_params())
