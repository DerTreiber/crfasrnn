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

    def hybrid_forward(self, F, x):
        return x.slice(begin=self.begin, end=self.end, step=self.step)

class Input(nn.HybridBlock):

    def __init__(self, **kwargs):
        super(Input, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return x

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

class MaxPool2DSamePadding(nn.HybridBlock):
    '''Expected to have the same output as tensorflow 'same' padding.

    Achieve 'same' padding same as in tensorflow keras:
    https://discuss.mxnet.io/t/pooling-and-convolution-with-same-mode/528/2
    Essentially: kernel_size=(k, k) -> padding=(k//2, k//2)
    If k is even, slice off first column and row.
    if k%2 == 0:
        pool = pool[:,:,1:,1:] <- replace this with slice operator
    '''

    def __init__(self, kernel_size, strides=(2, 2), **kwargs):
        self.kernel_size = kernel_size
        self.strides = strides
        super(MaxPool2DSamePadding, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(nn.MaxPool2D(self.kernel_size, strides=self.strides))

    def hybrid_forward(self, F, x):
        pool = self.net(x)
        if (self.kernel_size[0] % 2) == 0:
            pool = pool.slice(begin=(0,0,1,1), end=(None, None, None, None))
        return pool

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
    arr = np.random.rand(5, 3, 28, 28)

    xs = mx.nd.array(arr, dtype=np.float)
    h, w = xs.shape[:2]

    xs = xs.slice(begin=(0,0,1,1), end=(None, None, None, None))
    print(xs.shape)

    net = MaxPool2DSamePadding((2,2), strides=(2,2))

    net.hybridize()

    xs = net(xs)

    print(xs)