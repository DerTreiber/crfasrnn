"""
https://github.com/apache/incubator-mxnet/issues/8386
https://mxnet.incubator.apache.org/api/python/docs/api/ndarray/ndarray.html#mxnet.ndarray.slice
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping2D
"""

import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn


### emulate keras api as valid input: ((31, 37), (31, 37))
class CroppingLayer2D(nn.HybridBlock):

    def __init__(self, begin, end, step=(), **kwargs):
        super(CroppingLayer2D, self).__init__(**kwargs)
        self.begin = begin
        self.end = end
        self.step = step

    def forward(self, F, x):
        return x.slice(begin=self.begin, end=self.end, step=self.step)

class Input(nn.HybridBlock):

    def __init__(self, **kwargs):
        super(Input, self).__init__(**kwargs)

    def forward(self, input):
        return input

class Add(nn.HybridBlock):

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def forward(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output

class SequentialMultiInput(nn.HybridSequential):

    def hybrid_forward(self, F, *args):
        x = list(args)
        first = True
        for block in self._children.values():
            if first:
                x = block(*x)
                first = False
            else:
                x = block(x)
        return x

class ConcatLayer(nn.HybridBlock):

    def __init__(self, *args, **kwargs):
        super(ConcatLayer, self).__init__(**kwargs)
        self.layers = []
        for layer in args:
            self.register_child(layer)
            self.layers.append(layer)

    def hybrid_forward(self, F, *args):
        inputs = list(args)
        outputs = []
        for (layer, input) in zip(self.layers, inputs):
            outputs.append(layer(input))
        return F.concat(*outputs)

# class LogsDataset(Dataset):
#     def __init__(self):
#         self.len = int(1024)

#     def __getitem__(self, idx):
#         feature01 = nd.array(np.random.rand(1, 16, 16))
#         feature02 = nd.array(np.random.rand(100, 8))
#         feature03 = nd.array(np.random.rand(16))

#         label = nd.full(1, random.randint(0, 1), dtype="float32")

#         return feature01, feature02, feature03, label

#     def __len__(self):
#         return self.len

if __name__ == '__main__':
    # symbol_data = mx.sym.var('data')

    import numpy as np
    arr = np.random.rand(28, 28, 3)

    x = mx.nd.array(arr, dtype=np.float)

    # x = mx.nd.slice(data=x, begin=(2,4), end=(-2, -4))

    net = mx.gluon.nn.Sequential()

    conc = ConcatLayer()

    net1 = nn.Dense(10, activation='relu')
    net2 = nn.Dense(10, activation='relu')

    net1.initialize(mx.init.Normal(sigma=1.), ctx=mx.cpu())

    out1 = net1(x)
    out2 = net2(x)

    out = conc(out1, out2)
