import numpy as np
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn
mx.random.seed(1)

from crfrnn_layer import CrfRnnLayer
from custom_layers import CroppingLayer2D, Add


def get_crfrnn_model_def():
    """ Returns Keras CRN-RNN model definition.

    Currently, only 500 net 500 images are supported. However, one can get this to
    work with different image sizes by adjusting the parameters of the Cropping2D layers
    below.
    """

    channels, height, width = 3, 500, 500

    # Input
    input_shape = (height, width, channels)


    img_input = nn.Input(shape=input_shape)

    # with net.name_scope():
            
    ### maybe unnecessary
    # img_input = Input(shape=input_shape)

    # Add plenty of zero padding
    # net = ZeroPadding2D(padding=(100, 100))(img_input)

    # VGG-16 convolution block 1

    net = nn.Conv2D(64, (3, 3), activation='relu', padding=(100,100), name='conv1_1')
    print(net)
    net = nn.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(net)
    print(net)
    net = nn.MaxPool2D((2, 2), strides=(2, 2), name='pool1')(net)
    print(net)

    # VGG-16 convolution block 2
    net = nn.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(net)
    net = nn.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(net)
    net = nn.MaxPool2D((2, 2), strides=(2, 2), name='pool2', padding='same')(net)

    # VGG-16 convolution block 3
    net = nn.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(net)
    net = nn.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(net)
    net = nn.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(net)
    net = nn.MaxPool2D((2, 2), strides=(2, 2), name='pool3', padding='same')(net)
    pool3 = net

    # VGG-16 convolution block 4
    net = nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(net)
    net = nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(net)
    net = nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(net)
    net = nn.MaxPool2D((2, 2), strides=(2, 2), name='pool4', padding='same')(net)
    pool4 = net

    # VGG-16 convolution block 5
    net = nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(net)
    net = nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(net)
    net = nn.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(net)
    net = nn.MaxPool2D((2, 2), strides=(2, 2), name='pool5', padding='same')(net)

    # Fully-connected layers converted to convolution layers
    net = nn.Conv2D(4096, (7, 7), activation='relu', padding='valid', name='fc6')(net)
    net = nn.Dropout(0.5)(net)
    net = nn.Conv2D(4096, (1, 1), activation='relu', padding='valid', name='fc7')(net)
    net = nn.Dropout(0.5)(net)
    net = nn.Conv2D(21, (1, 1), padding='valid', name='score-fr')(net)

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

    output = CrfRnnLayer(image_dims=(height, width),
                        num_classes=21,
                        theta_alpha=160.,
                        theta_beta=3.,
                        theta_gamma=3.,
                        num_iterations=10,
                        name='crfrnn')([upscore, img_input])



    # # Build the model

    net = nn.Sequential(score_pool4)
    model = Model(img_input, output, name='crfrnn_net')

    return model
