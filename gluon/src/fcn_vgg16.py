import numpy as np
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn

from custom_layers import CroppingLayer2D, Add, Input


channels, height, width = 3, 500, 500

class FcnVGG16(nn.Block):
    def __init__(self, input_shape=(3,500,500), **kwargs):
        super(FcnVGG16, self).__init__(**kwargs)
        # Input
        self.input_shape = input_shape[::-1]

        ### maybe unnecessary
        # img_input = Input()

        # Add plenty of zero padding
        # net = ZeroPadding2D(padding=(100, 100))(img_input)

        # VGG-16 convolution block 1
        self.block1 = nn.Sequential()
        with self.block1.name_scope():
            # self.block1.add(img_input)
            self.block1.add(nn.Conv2D(64, (3, 3), activation='relu', padding=(100,100)))
            self.block1.add(nn.Conv2D(64, (3, 3), activation='relu'))
            self.block1.add(nn.MaxPool2D((2, 2), strides=(2, 2)))

        # VGG-16 convolution block 2
        self.block2 = nn.Sequential()
        with self.block2.name_scope():
            self.block2.add(self.block1)
            self.block2.add(nn.Conv2D(128, (3, 3), activation='relu'))
            self.block2.add(nn.Conv2D(128, (3, 3), activation='relu'))
            self.block2.add(nn.MaxPool2D((2, 2), strides=(2, 2)))

        # VGG-16 convolution block 3
        self.block3 = nn.Sequential()
        with self.block3.name_scope():
            self.block3.add(self.block2)
            self.block3.add(nn.Conv2D(256, (3, 3), activation='relu'))
            self.block3.add(nn.Conv2D(256, (3, 3), activation='relu'))
            self.block3.add(nn.Conv2D(256, (3, 3), activation='relu'))
            self.block3.add(nn.MaxPool2D((2, 2), strides=(2, 2)))

        # VGG-16 convolution block 4
        self.block4 = nn.Sequential()
        with self.block4.name_scope():
            self.block4.add(self.block3)
            self.block4.add(nn.Conv2D(512, (3, 3), activation='relu'))
            self.block4.add(nn.Conv2D(512, (3, 3), activation='relu'))
            self.block4.add(nn.Conv2D(512, (3, 3), activation='relu'))
            self.block4.add(nn.MaxPool2D((2, 2), strides=(2, 2)))

        # VGG-16 convolution block 5
        self.block5 = nn.Sequential()
        with self.block5.name_scope():
            self.block5.add(self.block4)
            self.block5.add(nn.Conv2D(512, (3, 3), activation='relu'))
            self.block5.add(nn.Conv2D(512, (3, 3), activation='relu'))
            self.block5.add(nn.Conv2D(512, (3, 3), activation='relu'))
            self.block5.add(nn.MaxPool2D((2, 2), strides=(2, 2)))

        # Fully-connected layers converted to convolution layers
        self.fc_end = nn.Sequential()
        with self.fc_end.name_scope():
            self.fc_end.add(self.block5)
            self.fc_end.add(nn.Conv2D(4096, (7, 7), activation='relu'))#, padding='valid'))
            self.fc_end.add(nn.Dropout(0.5))
            self.fc_end.add(nn.Conv2D(4096, (1, 1), activation='relu'))#, padding='valid'))
            self.fc_end.add(nn.Dropout(0.5))
            self.fc_end.add(nn.Conv2D(21, (1, 1)))#, padding='valid'))

        # Deconvolution
        self.deconv = nn.Sequential()
        with self.deconv.name_scope():
            self.deconv.add(self.fc_end)
            self.deconv.add(nn.Conv2DTranspose(21, (4, 4), strides=2))

        # Skip connections from pool4
        self.skip_4 = nn.Sequential()
        with self.skip_4.name_scope():
            self.skip_4.add(nn.Conv2D(21, (1, 1)))
            # score_pool4c = nn.Cropping2D((5, 5))(score_pool4)
            self.skip_4.add(CroppingLayer2D((5),(-5)))
            conc = gluon.contrib.nn.Concurrent(axis=-1)
            with conc.name_scope():
                conc.add(self.deconv)
                conc.add(self.skip_4)
            
            self.skip_4.add(conc)
            self.skip_4.add(nn.Conv2DTranspose(21, (4, 4), strides=2, use_bias=False))

        # Skip connections from pool3
        self.skip_3 = nn.Sequential()
        with self.skip_3.name_scope():
            self.skip_3.add(nn.Conv2D(21, (1, 1)))
            # score_pool3c = nn.Cropping2D((9, 9))(score_pool3)
            self.skip_3.add(CroppingLayer2D((9),(-9)))

            # # Fuse things together
            # score_final = Add()([score4, score_pool3c])
            conc2 = gluon.contrib.nn.Concurrent(axis=-1)
            with conc2.name_scope():
                conc2.add(self.skip_4)
                conc2.add(self.skip_3)


            # Final up-sampling and cropping
            # upsample = nn.Conv2DTranspose(21, (16, 16), strides=8, name='upsample', use_bias=False)(score4, score_pool3c)
            self.skip_3.add(nn.Conv2DTranspose(21, (16, 16), strides=8, use_bias=False))
            # upscore = nn.Cropping2D(((31, 37), (31, 37)))(upsample)
            self.skip_3.add(CroppingLayer2D((31, 31), (-37, -37)))

    def forward(self, X):
        out1 = self.deconv(X)
        out2 = self.skip_4(out1)
        out = self.skip_3(out2)
        return out

if __name__ == '__main__':
    model = FcnVGG16()

    symbol_data = mx.sym.var('data')
    tmp = model(symbol_data)