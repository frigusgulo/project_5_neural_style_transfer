"""VGG19 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend, layers, models
import os

import scipy.io

def VGG19(input_tensor=None, weight_path='imagenet-vgg-verydeep-19.mat',
          **kwargs):
    """Instantiates the VGG19 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    img_input = input_tensor
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Activation('relu',name='block1_relu1')(x)

    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.Activation('relu',name='block1_relu2')(x)

    x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Activation('relu',name='block2_relu1')(x)

    x = layers.Conv2D(128, (3, 3),
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.Activation('relu',name='block2_relu2')(x)

    x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Activation('relu',name='block3_relu1')(x)

    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Activation('relu',name='block3_relu2')(x)
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.Activation('relu',name='block3_relu3')(x)
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv4')(x)
    x = layers.Activation('relu',name='block3_relu4')(x)
    x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Activation('relu',name='block4_relu1')(x)
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Activation('relu',name='block4_relu2')(x)
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.Activation('relu',name='block4_relu3')(x)
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv4')(x)
    x = layers.Activation('relu',name='block4_relu4')(x)
    x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Activation('relu',name='block5_relu1')(x)
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Activation('relu',name='block5_relu2')(x)
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.Activation('relu',name='block5_relu3')(x)
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block5_conv4')(x)
    x = layers.Activation('relu',name='block5_relu4')(x)
    x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Create model.
    model = models.Model(img_input, x, name='vgg19')

    vgg_rawnet     = scipy.io.loadmat(weight_path)
    vgg_layers     = vgg_rawnet['layers'][0]

    layer_dict = {'block1_conv1': vgg_layers[0][0][0][2][0],
                  'block1_conv2': vgg_layers[2][0][0][2][0],
                  'block2_conv1': vgg_layers[5][0][0][2][0],
                  'block2_conv2': vgg_layers[7][0][0][2][0],
                  'block3_conv1': vgg_layers[10][0][0][2][0],
                  'block3_conv2': vgg_layers[12][0][0][2][0],
                  'block3_conv3': vgg_layers[14][0][0][2][0],
                  'block3_conv4': vgg_layers[16][0][0][2][0],
                  'block4_conv1': vgg_layers[19][0][0][2][0],
                  'block4_conv2': vgg_layers[21][0][0][2][0],
                  'block4_conv3': vgg_layers[23][0][0][2][0],
                  'block4_conv4': vgg_layers[25][0][0][2][0],
                  'block5_conv1': vgg_layers[28][0][0][2][0],
                  'block5_conv2': vgg_layers[30][0][0][2][0],
                  'block5_conv3': vgg_layers[32][0][0][2][0],
                  'block5_conv4': vgg_layers[34][0][0][2][0] }
    
    [model.get_layer(n).set_weights([w[0],w[1].squeeze()]) for n,w in layer_dict.items()]

    return model
