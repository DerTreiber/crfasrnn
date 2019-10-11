import numpy as np
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn
mx.random.seed(1)

model = nn.Sequential()



# Add additional convolutional layers to get the size down to at most 8x8
additional_layers = 0
while model.layers[-1].output_shape[2] > 8:
  additional_layers += 1
  layer = 5 + additional_layers
  model.add(nn.Conv2D(512, 3, 3, padding=(1,1), activation='relu',
                          name='conv%d_1' % layer))
  model.add(nn.Conv2D(512, 3, 3, padding=(1,1), activation='relu',
                          name='conv%d_2' % layer))
  model.add(nn.Conv2D(512, 3, 3, padding=(1,1), activation='relu',
                          name='conv%d_3' % layer))
  model.add(nn.MaxPool2D((2, 2), strides=(2, 2)))

  layer_size = model.layers[-1].output_shape[2]

# Add the fully convolutional layers
model.add(nn.Conv2D(4096, layer_size, layer_size, activation='relu',
                        name='fc6'))
model.add(nn.Dropout(0.5))
model.add(nn.Conv2D(4096, 1, 1, activation='relu', name='fc7'))
model.add(nn.Dropout(0.5))
model.add(nn.Conv2DTranspose(512, layer_size, layer_size,
                          (1, 512, layer_size, layer_size),
                          subsample=(2, 2), name='deconv-fc6'))

# Add deconvolutional layers for any additional convolutional layers
# that we added to get down to 8x8
for i in range(additional_layers, 0, -1):
  layer = 5 + i
  output_size = 2 * model.layers[-1].output_shape[2]
  model.add(UpSampling2D((2, 2)))
  model.add(nn.Conv2DTranspose(512, 3, 3,
                            (1, 512, output_size, output_size),
                            name='deconv%d_1' % layer,
                            border_mode='same'))
  model.add(nn.Conv2DTranspose(512, 3, 3,
                            (1, 512, output_size, output_size),
                            name='deconv%d_2' % layer,
                            border_mode='same'))
  model.add(nn.Conv2DTranspose(512, 3, 3,
                            (1, 512, output_size, output_size),
                            name='deconv%d_3' % layer,
                            border_mode='same'))

model.add(UpSampling2D((2, 2)))
output_size = model.layers[-1].output_shape[2]
model.add(nn.Conv2DTranspose(512, 3, 3, (1, 512, output_size, output_size),
                          border_mode='same', name='deconv5_1'))
model.add(nn.Conv2DTranspose(512, 3, 3, (1, 512, output_size, output_size),
                          border_mode='same', name='deconv5_2'))
model.add(nn.Conv2DTranspose(512, 3, 3, (1, 512, output_size, output_size),
                          border_mode='same', name='deconv5_3'))

model.add(UpSampling2D((2, 2)))
output_size = model.layers[-1].output_shape[2]
model.add(nn.Conv2DTranspose(512, 3, 3, (1, 512, output_size, output_size),
                          border_mode='same', name='deconv4_1'))
model.add(nn.Conv2DTranspose(512, 3, 3, (1, 512, output_size, output_size),
                          border_mode='same', name='deconv4_2'))
model.add(nn.Conv2DTranspose(256, 3, 3, (1, 256, output_size, output_size),
                          border_mode='same', name='deconv4_3'))

model.add(UpSampling2D((2, 2)))
output_size = model.layers[-1].output_shape[2]
model.add(nn.Conv2DTranspose(256, 3, 3, (1, 256, output_size, output_size),
                          border_mode='same', name='deconv3_1'))
model.add(nn.Conv2DTranspose(256, 3, 3, (1, 256, output_size, output_size),
                          border_mode='same', name='deconv3_2'))
model.add(nn.Conv2DTranspose(128, 3, 3, (1, 128, output_size, output_size),
                          border_mode='same', name='deconv3_3'))

model.add(UpSampling2D((2, 2)))
output_size = model.layers[-1].output_shape[2]
model.add(nn.Conv2DTranspose(128, 3, 3, (1, 128, output_size, output_size),
                          border_mode='same', name='deconv2_1'))
model.add(nn.Conv2DTranspose(64, 3, 3, (1, 64, output_size, output_size),
                          border_mode='same', name='deconv2_2'))

model.add(UpSampling2D((2, 2)))
output_size = model.layers[-1].output_shape[2]
model.add(nn.Conv2DTranspose(64, 3, 3, (1, 64, output_size, output_size),
                          border_mode='same', name='deconv1_1'))
model.add(nn.Conv2DTranspose(64, 3, 3, (1, 64, output_size, output_size),
                          border_mode='same', name='deconv1_2'))

model.add(nn.Conv2D(num_classes, 1, 1, activation='relu', name='output'))