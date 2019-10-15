"""
https://github.com/apache/incubator-mxnet/issues/8386
https://mxnet.incubator.apache.org/api/python/docs/api/ndarray/ndarray.html#mxnet.ndarray.slice
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Cropping2D
"""

import mxnet as mx
from mxnet import nd
from mxnet.gluon.nn import Block


class CroppingLayer2D(Block):
    def __init__(self, factor, **kwargs):
        super(CroppingLayer2D, self).__init__(**kwargs)
        self.factor = factor

    def forward(self, x):
        return mx.nd.slice(data=x, step=self.factor)

class Add(Block):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output
