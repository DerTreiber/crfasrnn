# fcn_test

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.data.vision import MNIST
from mxnet.gluon import nn

print(MNIST)
MNIST.transform

import gluoncv

### TODO:
### problem, does not include VGG16 backbone, which is used in original crfasrnn

# model = gluoncv.model_zoo.get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
# model.summary()



net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128, activation='relu'))
    net.add(nn.Dense(64, activation='relu'))
    net.add(nn.Dense(10))

### get last layer
print(net._children.items()[-1])

### iterate through layers
for key, layer in net._children.items():
    # print(key, layer)
    print(layer.collect_params())


model = nn.Sequential()


# Add additional convolutional layers to get the size down to at most 8x8
additional_layers = 0

# while model.layers[-1].output_shape[2] > 8:
additional_layers += 1
layer = 5 + additional_layers
model.add(nn.Conv2D(3, 3, 3, padding=(1,1), activation='relu'))
model.add(nn.Conv2D(3, 3, 3, padding=(1,1), activation='relu'))
model.add(nn.Conv2D(3, 3, 3, padding=(1,1), activation='relu'))
model.add(nn.MaxPool2D((2, 2), strides=(2, 2)))

# layer_size = model.layers[-1].output_shape[2]


for key, layer in model._children.items():
    # print(key, layer)
    print(layer.collect_params())

  

# model2 = gluoncv.model_zoo.vgg16()
# print(model2)
# combined = gluon.nn.Sequential()
# combined.add(model)
# combined.add(model2)
# print(combined)
