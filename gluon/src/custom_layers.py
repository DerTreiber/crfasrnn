"""
https://github.com/apache/incubator-mxnet/issues/8386
https://mxnet.incubator.apache.org/api/python/docs/api/ndarray/ndarray.html#mxnet.ndarray.slice
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping2D
"""

import mxnet as mx
from mxnet import nd
from mxnet.gluon.nn import Block


### emulate keras api as valid input: ((31, 37), (31, 37))
class CroppingLayer2D(Block):
    def __init__(self, begin, end, step=(), **kwargs):
        super(CroppingLayer2D, self).__init__(**kwargs)
        self.begin = begin
        self.end = end
        self.step = step

    def forward(self, x):
        return x.slice(begin=self.begin, end=self.end, step=self.step)

class Input(Block):
    def __init__(self, **kwargs):
        super(Input, self).__init__(**kwargs)

    def forward(self, input):
        return input

class Add(Block):
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
    with net.name_scope():
        net.add(CroppingLayer2D((2,4),(-2,-4)))
    
    x = net(x)
    print(x)