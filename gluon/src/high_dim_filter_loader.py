'''
https://mxnet.incubator.apache.org/api/faq/new_op
'''

import os
import mxnet as mx
from mxnet import nd
from mxnet.test_utils import get_mnist_iterator
import numpy as np
import logging


### TODO set attributes

@mx.operator.register('HighDimFilter')
class _high_dim_filter_grad(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        return 42

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        return 42

@mx.operator.register("HighDimFilter")
class _high_dim_filter_gradProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(_high_dim_filter_gradProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data']

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


### TODO replace load_op_library
# custom_module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'cpp', 'high_dim_filter.so'))

# @ops.RegisterGradient('HighDimFilter')
# def _high_dim_filter_grad(op, grad):
#     """ Gradients for the HighDimFilter op. We only need to calculate the gradients
#     w.r.t. the first input (unaries) as we never need to backprop errors to the
#     second input (RGB values of the image).

#     Args:
#     op: The `high_dim_filter` operation that we are differentiating.
#     grad: Gradients with respect to the output of the `high_dim_filter` op.

#     Returns:
#     Gradients with respect to the input of `high_dim_filter`.
#     """

#     rgb = op.inputs[1]
#     grad_vals = custom_module.high_dim_filter(grad, rgb,
#                                               bilateral=op.get_attr('bilateral'),
#                                               theta_alpha=op.get_attr('theta_alpha'),
#                                               theta_beta=op.get_attr('theta_beta'),
#                                               theta_gamma=op.get_attr('theta_gamma'),
#                                               backwards=True)

#     ### TODO replace tf.zeros_like
#     return [grad_vals, tf.zeros_like(rgb)]

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