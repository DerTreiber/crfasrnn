import numpy as np
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn
mx.random.seed(1)

import gluoncv


if mx.test_utils.list_gpus():
    print('Using GPU')
    print(mx.test_utils.list_gpus())
    ctx = mx.gpu()
else:
    ctx = mx.cpu()
    
### load training data
batch_size = 64

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(nn.Conv2D(channels=channels, kernel_size=3,
                      padding=1, activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out

def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out

# TODO write deconv_block and deconv_stack correctly
def deconv_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(nn.Conv2DTranspose(channels=channels, kernel_size=3,
                      padding=1, activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out

def deconv_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out

# TODO
# define multi_stage_mean_field as stack of cnn layers inside a RNN
def multi_stage_mean_field(num_iterations=10, compatibility_mode='POTTS', threshold=2, theta_alpha=160, theta_beta=3, theta_gamma=3, spatial_filter_weight=3, bilateral_filter_weight=5):
    out = nn.Sequential()
    print(42)

    return out

def fcn():
    out = nn.Sequential()

    return out



num_outputs = 10
architecture = ((1,64), (1,128), (2,256), (2,512))
net = nn.Sequential()
with net.name_scope():
    ### start FCN-VGG16
    net.add(vgg_stack(architecture))
    # TODO 
    # rewrite fully connected layers as convolutional layers

    # net.add(nn.Flatten())
    # net.add(nn.Dense(512, activation="relu"))
    # net.add(nn.Dropout(.5))
    # net.add(nn.Dense(512, activation="relu"))
    # net.add(nn.Dropout(.5))
    # net.add(nn.Dense(num_outputs))

    # TODO 
    # how to write skip connections
    # https://discuss.mxnet.io/t/end-to-end-gluon-model-with-skip-connections/3171/3

    net.add(deconv_stack(architecture[::-1]))

    ### end FCN-VGG16

    # TODO
    # net.add(multi_stage_mean_field())


net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .05})

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


###########################
#  Only one epoch so tests can run quickly, increase this variable to actually run
###########################
epochs = 1
smoothing_constant = .01

for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

        if i > 0 and i % 200 == 0:
            print('Batch %d. Loss: %f' % (i, moving_loss))

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))