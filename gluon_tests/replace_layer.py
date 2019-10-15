def replace_conv2D(net):
    for key, layer in net._children.items():
        if isinstance(layer, gluon.nn.Conv2D):
            new_conv = gluon.nn.Conv2D(
                channels=layer._channels, 
                kernel_size=layer._kwargs['kernel'], 
                strides=layer._kwargs['stride'], 
                padding=layer._kwargs['pad'], 
                use_bias=True)
            with net.name_scope():
                net.register_child(new_conv, key)
            new_conv.initialize(mx.init.Xavier())
            print('Replacing layer')

        # Recursively replace layers
        else:
            replace_conv2D(layer)
