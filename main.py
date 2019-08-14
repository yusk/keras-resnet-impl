from model import ResnetBuilder

if __name__ == "__main__":
    input_shape = (224, 224, 3)  # モデルの入力サイズ
    num_classes = 10  # クラス数

    # モデルを作成する。
    model = ResnetBuilder.build_resnet_34(input_shape, num_classes)

    # モデルをプロットする。
    from keras.utils import plot_model
    plot_model(model,
               to_file='resnet-model.png',
               show_shapes=True,
               show_layer_names=True)

    model.summary()
