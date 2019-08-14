from functools import reduce

from keras import backend as K
from keras.layers import (Activation, Add, BatchNormalization, Conv2D)
from keras.regularizers import l2


def compose(*funcs):
    '''複数の層を結合する。
    '''
    if funcs:
        return reduce(
            lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def ResNetConv2D(*args, **kwargs):
    '''conv を作成する。
    '''
    conv_kwargs = {
        'strides': (1, 1),
        'padding': 'same',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(1.e-4)
    }
    conv_kwargs.update(kwargs)

    return Conv2D(*args, **conv_kwargs)


def bn_relu_conv(*args, **kwargs):
    '''batch mormalization -> ReLU -> conv を作成する。
    '''
    return compose(BatchNormalization(), Activation('relu'),
                   ResNetConv2D(*args, **kwargs))


def shortcut(x, residual):
    '''shortcut connection を作成する。
    '''
    x_shape = K.int_shape(x)
    residual_shape = K.int_shape(residual)

    if x_shape == residual_shape:
        # x と residual の形状が同じ場合、なにもしない。
        shortcut = x
    else:
        # x と residual の形状が異なる場合、線形変換を行い、形状を一致させる。
        stride_w = int(round(x_shape[1] / residual_shape[1]))
        stride_h = int(round(x_shape[2] / residual_shape[2]))

        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_w, stride_h),
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1.e-4))(x)
    return Add()([shortcut, residual])


def basic_block(filters, first_strides, is_first_block_of_first_layer):
    '''bulding block を作成する。

        Arguments:
            filters: フィルター数
            first_strides: 最初の畳み込みのストライド
            is_first_block_of_first_layer: max pooling 直後の residual block かどうか
    '''
    def f(x):
        if is_first_block_of_first_layer:
            # conv1 で batch normalization -> ReLU はすでに適用済みなので、
            # max pooling の直後の residual block は畳み込みから始める。
            conv1 = ResNetConv2D(filters=filters,
                                 kernel_size=(3, 3))(x)  # 3x3_conv_64
        else:
            conv1 = bn_relu_conv(filters=filters,
                                 kernel_size=(3, 3),
                                 strides=first_strides)(x)  # 3x3_conv_64_/2
        conv2 = bn_relu_conv(filters=filters,
                             kernel_size=(3, 3))(conv1)  # 3x3_conv_64

        return shortcut(x, conv2)

    return f


def bottleneck_block(filters, first_strides, is_first_block_of_first_layer):
    '''bottleneck bulding block を作成する。

        Arguments:
            filters: フィルター数
            first_strides: 最初の畳み込みのストライド
            is_first_block_of_first_layer: max pooling 直後の residual block かどうか
    '''
    def f(x):
        if is_first_block_of_first_layer:
            # conv1 で batch normalization -> ReLU はすでに適用済みなので、
            # max pooling の直後の residual block は畳み込みから始める。
            conv1 = ResNetConv2D(filters=filters, kernel_size=(1, 1))(x)
        else:
            conv1 = bn_relu_conv(filters=filters,
                                 kernel_size=(1, 1),
                                 strides=first_strides)(x)

        conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        conv3 = bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv2)

        return shortcut(x, conv3)

    return f


def residual_blocks(block_function, filters, repetitions, is_first_layer):
    '''residual block を反復する構造を作成する。

        Arguments:
            block_function: residual block を作成する関数
            filters: フィルター数
            repetitions: residual block を何個繰り返すか。
            is_first_layer: max pooling 直後かどうか
    '''
    def f(x):
        for i in range(repetitions):
            # conv3_x, conv4_x, conv5_x の最初の畳み込みは、
            # プーリング目的の畳み込みなので、strides を (2, 2) にする。
            # ただし、conv2_x の最初の畳み込みは直前の max pooling 層でプーリングしているので
            # strides を (1, 1) にする。
            first_strides = (2, 2) if i == 0 and not is_first_layer else (1, 1)

            x = block_function(
                filters=filters,
                first_strides=first_strides,
                is_first_block_of_first_layer=(i == 0 and is_first_layer))(x)
        return x

    return f
