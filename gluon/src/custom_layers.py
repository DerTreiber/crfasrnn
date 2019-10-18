"""
https://github.com/apache/incubator-mxnet/issues/8386
https://mxnet.incubator.apache.org/api/python/docs/api/ndarray/ndarray.html#mxnet.ndarray.slice
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping2D
"""

import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn


### emulate keras api as valid input: ((31, 37), (31, 37))
class CroppingLayer2D(nn.Block):
    def __init__(self, begin, end, step=(), **kwargs):
        super(CroppingLayer2D, self).__init__(**kwargs)
        self.begin = begin
        self.end = end
        self.step = step

    def forward(self, x):
        return x.slice(begin=self.begin, end=self.end, step=self.step)

class ConcatLayer(nn.Block):
    def __init__(self, **kwargs):
        super(ConcatLayer, self).__init__(**kwargs) # attention here to pass kwargs to initialization of hybridblock

    def hybrid_forward(self, F, input_1, input_2): # You don't really need *args, **kwargs in this case
        result = F.concat(input_1, input_2, dim=1)
        return result

class Input(nn.Block):
    def __init__(self, **kwargs):
        super(Input, self).__init__(**kwargs)

    def forward(self, input):
        return input

class Add(nn.Block):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def forward(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output

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
    