import numpy as np
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn
mx.random.seed(1)

from crfrnn_layer import CrfRnnLayer
from custom_layers import CroppingLayer2D, Add, Input


def get_crfrnn_model_def():
    """ Returns Keras CRN-RNN model definition.

    Currently, only 500 net 500 images are supported. However, one can get this to
    work with different image sizes by adjusting the parameters of the Cropping2D layers
    below.
    """

    channels, height, width = 3, 500, 500

    # Input
    input_shape = (height, width, channels)


    # with net.name_scope():
            
    ### maybe unnecessary
    img_input = Input(shape=input_shape)

    # Add plenty of zero padding
    # net = ZeroPadding2D(padding=(100, 100))(img_input)

    # VGG-16 convolution block 1
    block1 = nn.Sequential()
    with block1.name_scope():
        block1.add(img_input)
        block1.add(nn.Conv2D(64, (3, 3), activation='relu', padding=(100,100), name='conv1_1'))
        block1.add(nn.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'))
        block1.add(nn.MaxPool2D((2, 2), strides=(2, 2), name='pool1'))

    # VGG-16 convolution block 2
    block2 = nn.Sequential()
    with block2.name_scope():
        block2.add(block1)
        block2.add(nn.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'))
        block2.add(nn.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'))
        block2.add(nn.MaxPool2D((2, 2), strides=(2, 2), name='pool2', padding='same'))

    # VGG-16 convolution block 3
    block3 = nn.Sequential()
    with block3.name_scope():
        block3.add(block2)
        block3.add(nn.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'))
        block3.add(nn.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'))
        block3.add(nn.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3'))
        block3.add(nn.MaxPool2D((2, 2), strides=(2, 2), name='pool3', padding='same'))

    # VGG-16 convolution block 4
    block4 = nn.Sequential()
    with block4.name_scope():
        block4.add(block3)
        block4.add(nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'))
        block4.add(nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'))
        block4.add(nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3'))
        block4.add(nn.MaxPool2D((2, 2), strides=(2, 2), name='pool4', padding='same'))

    # VGG-16 convolution block 5
    block5 = nn.Sequential()
    with block5.name_scope():
        block5.add(block4)
        block5.add(nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1'))
        block5.add(nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2'))
        block5.add(nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3'))
        block5.add(nn.MaxPool2D((2, 2), strides=(2, 2), name='pool5', padding='same'))

    # Fully-connected layers converted to convolution layers
    fc_end = nn.Sequential()
    with fc_end.name_scope():
        fc_end.add(block5)
        fc_end.add(nn.Conv2D(4096, (7, 7), activation='relu', padding='valid', name='fc6'))
        fc_end.add(nn.Dropout(0.5))
        fc_end.add(nn.Conv2D(4096, (1, 1), activation='relu', padding='valid', name='fc7'))
        fc_end.add(nn.Dropout(0.5))
        fc_end.add(nn.Conv2D(21, (1, 1), padding='valid', name='score-fr'))

    # Deconvolution
    score2 = nn.Conv2DTranspose(21, (4, 4), strides=2, name='score2')(net)

    # Skip connections from pool4
    score_pool4 = nn.Conv2D(21, (1, 1), name='score-pool4')(pool4)
    # score_pool4c = nn.Cropping2D((5, 5))(score_pool4)
    score_pool4c = CroppingLayer2D((5,5))(score_pool4)
    score_fused = Add()([score2, score_pool4c])
    score4 = nn.Conv2DTranspose(21, (4, 4), strides=2, name='score4', use_bias=False)(score_fused)

    # Skip connections from pool3
    score_pool3 = nn.Conv2D(21, (1, 1), name='score-pool3')(pool3)
    # score_pool3c = nn.Cropping2D((9, 9))(score_pool3)
    score_pool3c = CroppingLayer2D((9,9))(score_pool3)

    # # Fuse things together
    score_final = Add()([score4, score_pool3c])

    # Final up-sampling and cropping
    # upsample = nn.Conv2DTranspose(21, (16, 16), strides=8, name='upsample', use_bias=False)(score4, score_pool3c)
    upsample = nn.Conv2DTranspose(21, (16, 16), strides=8, name='upsample', use_bias=False)(score_final)
    # upscore = nn.Cropping2D(((31, 37), (31, 37)))(upsample)
    upscore = CroppingLayer2D(((31, 37), (31, 37)))(upsample)

    output = nn.Sequential()
    with output.name_scope():
        output.add(deconv)
        output.add(CrfRnnLayer(image_dims=(height, width),
                            num_classes=21,
                            theta_alpha=160.,
                            theta_beta=3.,
                            theta_gamma=3.,
                            num_iterations=10,
                            name='crfrnn')([upscore, img_input]))



    # # Build the model

    net = nn.Sequential(score_pool4)
    with net.name_scope():
        net.add(output)

    return  net
