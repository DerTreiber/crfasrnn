# my TODOs

this is just a personal todo list and collection of more or less relevant notes

## fcn_vgg16

- implemented prelimary network, network graph looks fine
- test performance of fcn
- understand how skip connections work
- verify correct adaption from keras model, especially padding in conv layers
- train model/use pretrained weights from shelhammer et al. to make a demo

## crfrnn

### implement crfrnn layer in mxnet/gluon

- define coarse structure
- possibly rewrite working steps as seperate layers

### permutohedral lattice

- look up if implementation of permutohedral lattice filter in gluon already exists, which doesn't seem the case
- python implementation found at: https://github.com/idofr/pymutohedral_lattice
- adapt this python implementation as custom operator for mxnet
- current problem: understanding how to make bilateral filters, passing theta arguments like in the original crf implementation
- python custom mxnet operator should be just a temporary workaround, in the future a custom mxnet operator in cpp should replace it
- python implementation very slow, not making use of gpu

## next steps:

- combine models
- define data loader
- train model (at least with subset)


## next steps: emadl

- write sample network in EMADL
- sketch/write fcn_vgg16 in EMADL
- determine what functionalities might be missing in EMADL2GLUON
- sketch/write crfrnn layer in EMADL

## other notes:

- when rendering a mxnet model graph activate a conda env with graphviz installed
- /usr/bin/python doesn't seem to work
- understand what the difference between mxnet nd and symbol is
- understand permutohedral lattice and high dimensional filtering