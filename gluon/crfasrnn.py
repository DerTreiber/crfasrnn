import numpy as np
import mxnet as mx
from mxnet import gluon


ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx

net = gluon.nn.Sequential()
#with net.name_scope():
    