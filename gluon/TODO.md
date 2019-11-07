# my TODOs

this is just a personal todo list and collection of more or less relevant notes

## fcn_vgg16

- implemented prelimary network, network graph looks fine
- make graph more readable, add names to operations where possible, turn by 90°
- test performance of fcn
- understand how skip connections work
- verify correct adaption from keras model, especially padding in conv layers
- train model/use pretrained weights from shelhammer et al. to make a demo

## crfrnn

### implement crfrnn layer in mxnet/gluon

- define coarse structure
- possibly rewrite working steps as seperate layers

### permutohedral lattice

- ~~look up if implementation of permutohedral lattice filter in gluon already exist~~ doesn't seem the case
- python implementation of permutohedral lattice filter found at: https://github.com/idofr/pymutohedral_lattice
- adapt this python implementation as custom operator for mxnet
- current problem: understanding how to make bilateral filters, adjusting implementation to account for theta variables
- python custom mxnet operator should be just a temporary workaround, in the future a custom mxnet operator in cpp should replace it
- python implementation very slow in comparison to cpp implementation
- backwards pass needs to be implemented. error gradients are computed by passing the error through the same M filters in reverse order,
which means in terms of permutohedral lattice operations to switch filters in the blur stage, while keeping the rest (building the permutohedral
lattice, the splatting, and splicing) the same as in the forward pass (refer to CRF-RNN paper section 4.2)
- this is achieved by setting the reverse flag in the blur function

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