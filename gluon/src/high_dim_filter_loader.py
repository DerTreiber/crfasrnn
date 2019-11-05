'''
https://mxnet.incubator.apache.org/api/faq/new_op
'''

# import sys
# sys.path.insert(0, 'pymutohedral_lattice')

import os
import mxnet as mx
from mxnet import nd
from mxnet.test_utils import get_mnist_iterator
import numpy as np
import logging
from pymutohedral_lattice.permutohedral_lattice import PermutohedralLattice


### TODO set attributes

@mx.operator.register('HighDimFilter')
class _high_dim_filter_grad(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        ### TODO do permutohedral stuff
        ###
        y = call_permutohedral_lattice()
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        l = in_data[1].asnumpy().ravel().astype(np.int)
        y = out_data[0].asnumpy()
        ### TODO do permutohedral stuff
        ###

        self.assign(in_grad[0], req[0], mx.nd.array(y))

@mx.operator.register("HighDimFilter")
class _high_dim_filter_gradProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(_high_dim_filter_gradProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data',
                'bilateral',
                'theta_alpha',
                'theta_beta',
                'theta_gamma']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return _high_dim_filter_grad()

def call_permutohedral_lattice(im, bilateral=True):
    '''Calls python implementation of permutohedral lattice
    Reference: implementation in https://github.com/idofr/pymutohedral_lattice
    '''
    invSpatialStdev = float(1. / 5.)
    invColorStdev = float(1. / .125)

    ### TODO put constructor for image positions in crfrnn layer as it stays the same for fixed input
    # Construct the position vectors out of x, y, r, g, and b.
    positions = np.zeros((im.shape[0], im.shape[1], 5), dtype='float32')
    for r in range(im.shape[0]):
        for c in range(im.shape[1]):
            positions[r, c, 0] = invSpatialStdev * c
            positions[r, c, 1] = invSpatialStdev * r
            positions[r, c, 2] = invColorStdev * im[r, c, 0]
            positions[r, c, 3] = invColorStdev * im[r, c, 1]
            positions[r, c, 4] = invColorStdev * im[r, c, 2]

    out = PermutohedralLattice.filter(im, positions)
    return out

if __name__ == '__main__':
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    #mlp = mx.symbol.Softmax(data = fc3, name = 'softmax')
    mlp = mx.symbol.Softmax(data=fc3, name='softmax', op_type='softmax')

    # data

    train, val = get_mnist_iterator(batch_size=100, input_shape = (784,))

    # train

    logging.basicConfig(level=logging.DEBUG)

    # MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
    context=mx.cpu()
    # Uncomment this line to train on GPU
    # context=mx.gpu(0)

    mod = mx.mod.Module(mlp, context=context)

    mod.fit(train_data=train, eval_data=val, optimizer='sgd',
        optimizer_params={'learning_rate':0.1, 'momentum': 0.9, 'wd': 0.00001},
        num_epoch=10, batch_end_callback=mx.callback.Speedometer(100, 100))