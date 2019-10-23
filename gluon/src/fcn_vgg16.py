import numpy as np
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn

from custom_layers import CroppingLayer2D, Add, Input, ConcatLayer, SequentialMultiInput


channels, height, width = 3, 500, 500

class FcnVGG16(nn.HybridBlock):
    def __init__(self, input_shape=(3,500,500), **kwargs):
        super(FcnVGG16, self).__init__(**kwargs)
        # Input
        self.input_shape = input_shape[::-1]

        # VGG-16 convolution block 1
        self.block1 = nn.HybridSequential()
        with self.block1.name_scope():
            # self.block1.add(img_input)
            self.block1.add(nn.Conv2D(64, (3, 3), activation='relu', padding=(100,100)))
            self.block1.add(nn.Conv2D(64, (3, 3), activation='relu'))
            self.block1.add(nn.MaxPool2D((2, 2), strides=(2, 2)))

        # VGG-16 convolution block 2
        self.block2 = nn.HybridSequential()
        with self.block2.name_scope():
            self.block2.add(self.block1)
            self.block2.add(nn.Conv2D(128, (3, 3), activation='relu'))
            self.block2.add(nn.Conv2D(128, (3, 3), activation='relu'))
            self.block2.add(nn.MaxPool2D((2, 2), strides=(2, 2)))

        # VGG-16 convolution block 3
        self.block3 = nn.HybridSequential()
        with self.block3.name_scope():
            self.block3.add(self.block2)
            self.block3.add(nn.Conv2D(256, (3, 3), activation='relu'))
            self.block3.add(nn.Conv2D(256, (3, 3), activation='relu'))
            self.block3.add(nn.Conv2D(256, (3, 3), activation='relu'))
            self.block3.add(nn.MaxPool2D((2, 2), strides=(2, 2)))

        # VGG-16 convolution block 4
        self.block4 = nn.HybridSequential()
        with self.block4.name_scope():
            # self.block4.add(self.block3)
            self.block4.add(nn.Conv2D(512, (3, 3), activation='relu'))
            self.block4.add(nn.Conv2D(512, (3, 3), activation='relu'))
            self.block4.add(nn.Conv2D(512, (3, 3), activation='relu'))
            self.block4.add(nn.MaxPool2D((2, 2), strides=(2, 2)))

        # VGG-16 convolution block 5
        self.block5 = nn.HybridSequential()
        with self.block5.name_scope():
            # self.block5.add(self.block4)
            self.block5.add(nn.Conv2D(512, (3, 3), activation='relu'))
            self.block5.add(nn.Conv2D(512, (3, 3), activation='relu'))
            self.block5.add(nn.Conv2D(512, (3, 3), activation='relu'))
            self.block5.add(nn.MaxPool2D((2, 2), strides=(2, 2)))

        # Fully-connected layers converted to convolution layers
        self.fc_end = nn.HybridSequential()
        with self.fc_end.name_scope():
            self.fc_end.add(self.block5)
            self.fc_end.add(nn.Conv2D(4096, (7, 7), activation='relu'))#, padding='valid'))
            self.fc_end.add(nn.Dropout(0.5))
            self.fc_end.add(nn.Conv2D(4096, (1, 1), activation='relu'))#, padding='valid'))
            self.fc_end.add(nn.Dropout(0.5))
            self.fc_end.add(nn.Conv2D(21, (1, 1)))#, padding='valid'))

        # Deconvolution
        self.deconv = nn.HybridSequential()
        with self.deconv.name_scope():
            self.deconv.add(self.fc_end)
            self.deconv.add(nn.Conv2DTranspose(21, (4, 4), strides=2))

        # Skip connections from pool4
        self.skip_4 = nn.HybridSequential()
        with self.skip_4.name_scope():
            # self.skip_4.add(self.block4)
            self.skip_4.add(nn.Conv2D(21, (1, 1)))
            self.skip_4.add(CroppingLayer2D((5),(-5)))
        
        self.conc1 = SequentialMultiInput()
        with self.conc1.name_scope():
            # self.conc1.add(ConcatLayer(self.deconv, self.skip_4))
            
            self.conc1.add(nn.Conv2DTranspose(21, (4, 4), strides=2, use_bias=False))

        # Skip connections from pool3
        self.skip_3 = nn.HybridSequential()
        with self.skip_3.name_scope():
            self.skip_3.add(self.block3)
            self.skip_3.add(nn.Conv2D(21, (1, 1)))
            # score_pool3c = nn.Cropping2D((9, 9))(score_pool3)
            self.skip_3.add(CroppingLayer2D((9),(-9)))
        
        self.out = SequentialMultiInput()
        with self.out.name_scope():
            # self.out.add(ConcatLayer(self.skip_4, self.conc1))

            # Final up-sampling and cropping
            self.out.add(nn.Conv2DTranspose(21, (16, 16), strides=8, use_bias=False))
            self.out.add(CroppingLayer2D((31, 31), (-37, -37)))

    def hybrid_forward(self, F, X):
        out_block1_3 = self.block3(X)
        out_block4 = self.block4(out_block1_3)
        out_block5_fc_deconv = self.deconv(out_block4)
        out_skip4 = self.skip_4(out_block4)
        out_skip3 = self.skip_3(out_block5_fc_deconv)
        out_conc1 = self.conc1(out_block5_fc_deconv + out_skip4)
        out = self.out(out_skip3 + out_conc1)

        return out

if __name__ == '__main__':
    model = FcnVGG16()
    model.hybridize()

    symbol_data = mx.sym.var('data')
    print(type(symbol_data))

    tmp = model(symbol_data)

    digraph = mx.viz.plot_network(tmp, title='fcn_mxnet')
    digraph.view(filename='fcn_mxnet')
