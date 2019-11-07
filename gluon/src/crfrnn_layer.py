'''
https://mxnet.incubator.apache.org/api/faq/new_op
https://github.com/apache/incubator-mxnet/blob/master/example/numpy-ops/custom_softmax.py
'''

import numpy as np
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn

import high_dim_filter_loader


def _diagonal_initializer(shape):
    return nd.array(np.eye(shape[0], shape[1], dtype=np.float32))


def _potts_model_initializer(shape):
    return nd.array(-1 * _diagonal_initializer(shape))


### TODO encapsulate single steps of mean-field algorithm as layers

class CrfRnnLayer(nn.HybridBlock):
    """ Implements the CRF-RNN layer described in:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations, **kwargs):
        super(CrfRnnLayer, self).__init__(**kwargs)

        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        self.add_vars()

    def add_vars(self):
        '''
        https://discuss.mxnet.io/t/hybridblock-hybrid-forward-extra-arguments-for-parameters/1764/4?u=dertreiber
        https://discuss.mxnet.io/t/how-to-initialize-weights-of-conv2d-layer-given-an-ndarray/4442/2?u=dertreiber
        self.fc3_weight = self.params.get('fc3_weight', shape=(10, 84)).initialize(mx.init.Xavier())
        '''

        ### TODO write custom initializers

        shape = (self.num_classes, self.num_classes)

        # Weights of the spatial kernel
        self.spatial_ker_weights = self.params.get('spatial_ker_weights',
                                                   shape=shape
                                                   ).initialize(mx.init.Constant(_diagonal_initializer(shape)))


        # Weights of the bilateral kernel
        self.bilateral_ker_weights = self.params.get('bilateral_ker_weights',
                                                     shape=shape
                                                     ).initialize(mx.init.Constant(_diagonal_initializer(shape)))

        # Compatibility matrix
        self.compatibility_matrix = self.params.get('compatibility_matrix',
                                                    shape=shape
                                                    ).initialize(mx.init.Constant(_potts_model_initializer(shape)))


    def forward(self, F, x):
        ''' https://mxnet.incubator.apache.org/api/faq/new_op
            # mlp = mx.symbol.Custom(data=fc3, name='softmax', op_type='softmax')
        '''
        ### transform inputs
        unaries = F.transpose(x[0][0, :, :, :], axes=(2, 0, 1))
        rgb = F.transpose(x[1][0, :, :, :], axes=(2, 0, 1))

        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
        all_ones = np.ones((c, h, w), dtype=np.float32)

        # Prepare filter normalization coefficients
        # spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
        #                                                   theta_gamma=self.theta_gamma)

        spatial_norm_vals = F.Custom(op=all_ones, data=rgb,
                                     name='HighDimFilter', op_type='HighDimFilter',
                                     bilateral=False,
                                     theta_gamma=self.theta_gamma)

        # bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
        #                                                     theta_alpha=self.theta_alpha,
        #                                                     theta_beta=self.theta_beta)
        bilateral_norm_vals = F.Custom(op=all_ones, data=rgb,
                                       name='HighDimFilter', op_type='HighDimFilter',
                                       bilateral=True,
                                       theta_alpha=self.theta_alpha, theta_beta=self.theta_beta)
        q_values = unaries

        for i in range(self.num_iterations):
            # softmax_out = F.softmax(q_values)

            # Spatial filtering
            spatial_out = F.Custom(op=all_ones, data=rgb,
                                        name='HighDimFilter', op_type='HighDimFilter',
                                        bilateral=False,
                                        theta_gamma=self.theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals

            # Bilateral filtering
            bilateral_out = F.Custom(op=all_ones, data=rgb,
                                     name='HighDimFilter', op_type='HighDimFilter',
                                     bilateral=True,
                                     theta_alpha=self.theta_alpha, theta_beta=self.theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals

            ### TODO check for correct dimensions in dot multiplication
            # Weighting filter outputs
            message_passing = (F.batch_dot(self.spatial_ker_weights,
                                         F.reshape(spatial_out, (c, -1))) +
                               F.batch_dot(self.bilateral_ker_weights,
                                         F.reshape(bilateral_out, (c, -1))))

            # Compatibility transform
            pairwise = F.batch_dot(self.compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = F.reshape(pairwise, (c, h, w))
            q_values = unaries - pairwise

        return F.transpose(F.reshape(q_values, (1, c, h, w)), axes=(0, 2, 3, 1))

    def compute_output_shape(self, input_shape):
        return input_shape

# test_arr = nd.array([[[ 1,  2,  3],[ 4,  5,  6]],[[ 7,  8,  9],[10, 11, 12]]], ctx=None)

if __name__ == '__main__':
    print(mx.init.Constant(1))