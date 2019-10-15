import mxnet as mx 
from mxnet import nd, gluon

class ConcatLayer(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        gluon.nn.HybridBlock.__init__(self, **kwargs) # attention here to pass kwargs to initialization of hybridblock
        with self.name_scope():
            self.a = gluon.nn.Dense(3)
            self.b = gluon.nn.Dense(5)

    def hybrid_forward(self, F, input_1, input_2): # You don't really need *args, **kwards in this case
        out1 = self.a(input_1)
        out2 = self.b(input_2)

        result = F.concat(out1, out2, dim=1)
        return result


net = ConcatLayer() # You don't need Sequential or HybridSequential for a single layer
net.hybridize(static_alloc=True,static_shape=True) # lightning gluon speed :) 
some_ctx = mx.cpu() # modified thisf rom your code 
net.initialize(mx.init.Normal(sigma=1.), ctx=some_ctx) # you don't need collect_params anymore for initializing, change in ctx definition

batch = 32
input1 = nd.random.uniform(shape = [batch, 10])
input2 = nd.random.uniform(shape = [batch, 20])

out = net(input1,input2)
print (out.shape) 
#(prints (32,8))