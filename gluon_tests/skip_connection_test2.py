from mxnet import gluon, nd, autograd
from mxnet.gluon.nn import HybridSequential, HybridBlock, Dense
from mxnet.gluon import Trainer
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
import mxnet as mx
from mxnet.gluon.data import DataLoader, Dataset

import numpy as np
import random


class SequentialMultiInput(HybridSequential):

    def hybrid_forward(self, F, *args):
        x = list(args)
        first = True
        for block in self._children.values():
            if first:
                x = block(*x)
                first = False
            else:
                x = block(x)
        return x

class ConcatLayer(HybridBlock):

    def __init__(self, *args, **kwargs):
        super(ConcatLayer, self).__init__(**kwargs)
        self.layers = []
        for layer in args:
            self.register_child(layer)
            self.layers.append(layer)

    def hybrid_forward(self, F, *args):
        inputs = list(args)
        outputs = []
        for (layer, input) in zip(self.layers, inputs):
            outputs.append(layer(input))
        return F.concat(*outputs)

net = SequentialMultiInput("merge_")
with net.name_scope():
    net.add(ConcatLayer(Dense(3), Dense(4), Dense(5)))
    net.add(Dense(2))
net.hybridize()

class LogsDataset(Dataset):
    def __init__(self):
        self.len = int(1024)

    def __getitem__(self, idx):
        feature01 = nd.array(np.random.rand(1, 16, 16))
        feature02 = nd.array(np.random.rand(100, 8))
        feature03 = nd.array(np.random.rand(16))

        label = nd.full(1, random.randint(0, 1), dtype="float32")

        return feature01, feature02, feature03, label

    def __len__(self):
        return self.len

train_data = DataLoader(LogsDataset(), batch_size=64)

ctx = mx.cpu()
net.initialize()
softmax_loss = SoftmaxCrossEntropyLoss()
trainer = Trainer(net.collect_params(), optimizer="adam")

for epoch in range(5):
    for idx, (X01, X02, X03, y) in enumerate(train_data):
        with autograd.record():
            output = net(X01, X02, X03)
            loss = softmax_loss(output, y)

        loss.backward()
        trainer.step(64)

net.export("net")