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
    def __init__(self, bilateral=True, theta_alpha=None, theta_beta=None, theta_gamma=None):
        super(_high_dim_filter_grad, self).__init__()

        self.bilateral = bilateral
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma


    def forward(self, is_train, req, in_data, out_data, aux):
        ### TODO write as mxnet operators
        ### conversion to numpy for convenience
        y = in_data[0].asnumpy()

        if self.bilateral:
            positions = compute_bilateral_kernel(y, self.theta_alpha, self.theta_beta)
        elif not self.bilateral:
            positions = compute_spatial_kernel(y, self.theta_gamma)

        y = PermutohedralLattice.filter(y, positions)
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = out_data[0].asnumpy()
        ### TODO do permutohedral stuff
        ### backward pass: pass error through the same M gaussian filters in reverse order
        ### in terms of permutohedral lattice operations this can be accomplished by reversing the order in the blur stage,
        ### while keeping the order for building the permutohedral lattice, splatting, and slicing stays the same
        ### as in the forward pass.


        if self.bilateral:
            positions = compute_bilateral_kernel(y, self.theta_alpha, self.theta_beta)
        elif not self.bilateral:
            positions = compute_spatial_kernel(y, self.theta_gamma)

        y = PermutohedralLattice.filter(y, positions, reverse=True)

        self.assign(in_grad[0], req[0], mx.nd.array(y))

@mx.operator.register("HighDimFilter")
class _high_dim_filter_gradProp(mx.operator.CustomOpProp):
    def __init__(self, bilateral=True, theta_alpha=None, theta_beta=None, theta_gamma=None):
        super(_high_dim_filter_gradProp, self).__init__(need_top_grad=False)

        self.bilateral = bilateral
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma

    def list_arguments(self):
        return ['data', 'label']

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
        return _high_dim_filter_grad(bilateral=self.bilateral, theta_alpha=self.theta_alpha, theta_beta=self.theta_beta, theta_gamma=self.theta_gamma)

def compute_spatial_kernel(im, theta_gamma):

    positions = np.zeros((im.shape[0], im.shape[1], 2), dtype='float32')
    for r in range(im.shape[0]):
        for c in range(im.shape[1]):
            positions[r, c, 0] = theta_gamma * c
            positions[r, c, 1] = theta_gamma * r

    return positions

def compute_bilateral_kernel(im, theta_alpha, theta_beta):

    positions = np.zeros((im.shape[0], im.shape[1], 5), dtype='float32')
    for r in range(im.shape[0]):
        for c in range(im.shape[1]):
            positions[r, c, 0] = theta_alpha * c
            positions[r, c, 1] = theta_alpha * r
            positions[r, c, 2] = theta_beta * im[r, c, 0]
            positions[r, c, 3] = theta_beta * im[r, c, 1]
            positions[r, c, 4] = theta_beta * im[r, c, 2]

    return positions


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