# fcn_test

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd

import gluoncv

### TODO:
### problem, does not include VGG16 backbone, which is used in original crfasrnn

model = gluoncv.model_zoo.get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
print(model)

model2 = gluoncv.model_zoo.vgg16()
# print(model2)
# combined = gluon.nn.Sequential()
# combined.add(model)
# combined.add(model2)
# print(combined)
