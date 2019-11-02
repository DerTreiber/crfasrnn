import numpy as np
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn
mx.random.seed(1)

from crfrnn_layer import CrfRnnLayer
from custom_layers import CroppingLayer2D, Add, Input
from fcn_vgg16 import FcnVGG16 as FCN


class SegCRFRNN(nn.HybridBlock):
    """ Returns Gluon CRN-RNN model definition.

    Currently, only 500 net 500 images are supported. However, one can get this to
    work with different image sizes by adjusting the parameters of the Cropping2D layers
    below.
    """

    def __init__(self, input_shape=(3, 500, 500), **kwargs):
        super(SegCRFRNN, self).__init__(**kwargs)

        self.input_shape = input_shape[::-1]

        self.fcn = FCN()

        self.crfrnn = nn.Sequential()
        with self.crfrnn.name_scope():
            self.crfrnn.add(CrfRnnLayer(image_dims=(input_shape[1], input_shape[0]),
                                        num_classes=21,
                                        theta_alpha=160.,
                                        theta_beta=3.,
                                        theta_gamma=3.,
                                        num_iterations=10,
                                        name='crfrnn'))

    def hybrid_forward(self, F, X, *args, **kwargs):
        unary_potentials = self.fcn(X)
        out = self.crfrnn(unary_potentials+X)
        return out

if __name__ == '__main__':
    model = SegCRFRNN()
    model.hybridize()

    symbol_data = mx.sym.var('data')

    tmp = model(symbol_data)

    digraph = mx.viz.plot_network(tmp, title='fcn_mxnet')
    digraph.view(filename='fcn_mxnet')
