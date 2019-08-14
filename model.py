from keras.models import Model
from keras.layers import (Activation, GlobalAveragePooling2D,
                          BatchNormalization, Dense, Input, MaxPooling2D)

from funcs import (basic_block, bottleneck_block, compose, residual_blocks,
                   ResNetConv2D)


class ResnetBuilder():
    @staticmethod
    def build(input_shape, num_outputs, block_type, repetitions):
        '''ResNet モデルを作成する Factory クラス

        Arguments:
            input_shape: 入力の形状
            num_outputs: ネットワークの出力数
            block_type : residual block の種類 ('basic' or 'bottleneck')
            repetitions: 同じ residual block を何個反復させるか
        '''
        # block_type に応じて、residual block を生成する関数を選択する。
        if block_type == 'basic':
            block_fn = basic_block
        elif block_type == 'bottleneck':
            block_fn = bottleneck_block

        # モデルを作成する。
        ##############################################
        input = Input(shape=input_shape)

        # conv1 (batch normalization -> ReLU -> conv)
        conv1 = compose(
            ResNetConv2D(filters=64,
                         kernel_size=(7, 7),
                         strides=(2, 2),
                         name="7x7_conv_64_/2"),
            BatchNormalization(),
            Activation('relu'),
        )(input)

        # pool
        pool1 = MaxPooling2D(pool_size=(3, 3),
                             strides=(2, 2),
                             padding='same',
                             name="pool_/2")(conv1)

        # conv2_x, conv3_x, conv4_x, conv5_x
        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = residual_blocks(block_fn,
                                    filters=filters,
                                    repetitions=r,
                                    is_first_layer=(i == 0))(block)
            filters *= 2

        # batch normalization -> ReLU
        block = compose(BatchNormalization(), Activation('relu'))(block)

        # global average pooling
        pool2 = GlobalAveragePooling2D(name="avg_pool")(block)

        # dense
        fc1 = Dense(units=num_outputs,
                    kernel_initializer='he_normal',
                    activation='softmax',
                    name="fc")(pool2)

        return Model(inputs=input, outputs=fc1)

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, 'basic',
                                   [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, 'basic',
                                   [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, 'bottleneck',
                                   [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, 'bottleneck',
                                   [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, 'bottleneck',
                                   [3, 8, 36, 3])