# fcn_test
import sys
sys.path.insert(0, 'gluon/src')

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon.data.vision import MNIST
from mxnet.gluon import nn
# from custom_layers import Add

net1 = nn.Sequential()

with net1.name_scope():
    net1.add(nn.Dense(3, activation='relu'))

net2 = nn.Sequential()

with net2.name_scope():
    net2.add(nn.Dense(3, activation='relu'))


net3 = nn.Sequential()

with net3.name_scope():
    # net3.add(nn.Conc)
    net3.add(nn.Dense(10, activation='relu'))
    # net3.add()

input = np.random.random((500,500,3))

nd_arr = nd.array([[20,20,10],[30,30,40]])
tmp1 = net1(nd_arr)
tmp2 = net2(nd_arr)
tmp3 = net3([tmp1, tmp2])
# tmp(block1)

digraph = mx.viz.plot_network(tmp3)
digraph.render()


# for key, layer in model._children.items():
#     # print(key, layer)
#     print(layer.collect_params())

