import numpy as np
import mxnet as mx
from mxnet.gluon import nn

class Example(nn.HybridBlock):
    def __init__(self, filters):
        super(Example, self).__init__()
        
        self.F1, self.F3 = filters
        with self.name_scope():
            self.part1 = nn.HybridSequential()
            with self.part1.name_scope():           
                self.part1.add(nn.Conv2D(channels=self.F1, kernel_size=(1,1), strides=(1,1)))
                self.part1.add(nn.BatchNorm(axis=1))
                self.part1.add(nn.Activation(activation='relu'))
        
                self.part1.add(nn.Conv2D(channels=self.F3,kernel_size=(1,1), strides=(1,1)))
                self.part1.add(nn.BatchNorm(axis=1))
            
    def hybrid_forward(self, F, X):  
        l1 = self.part1
        return (l1 + X).relu() 

if __name__ == '__main__':

    conv_block = Example(filters=[64, 256])
    symbol_data = mx.sym.var('data')
    tmp = conv_block(symbol_data)

    digraph = mx.viz.plot_network(tmp)
    digraph.render()
