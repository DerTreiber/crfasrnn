# my TODOs

this is just for notes and todo points

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
- implement permutohedral lattice filter or find temporary workaround

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