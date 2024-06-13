import keras
import keras_cv
import numpy as np
import tensorflow as tf


class CNN(object):
    def __init__(self, image_height, image_width, max_captcha, char_set, model_save_dir, dropout_rate):
        # 初始值
        self.image_height = image_height
        self.image_width = image_width
        self.max_captcha = max_captcha
        self.char_set = char_set
        self.char_set_len = len(char_set)
        self.model_save_dir = model_save_dir  # 模型路径

        self.dropout_rate = dropout_rate  # dropout值

    def text2vec(self, text):
        """
        转标签为oneHot编码
        :param text: str
        :return: numpy.array
        """
        text_len = len(text)
        if text_len > self.max_captcha:
            raise ValueError('验证码最长{}个字符'.format(self.max_captcha))

        vector = np.zeros(self.max_captcha * self.char_set_len)

        for i, ch in enumerate(text):
            idx = i * self.char_set_len + self.char_set.index(ch)
            vector[idx] = 1
        return vector

    # 自建模型
    def model(self):
        model = tf.keras.Sequential()  # 构建顺序模型
        # 输入层
        model.add(tf.keras.Input(shape=(self.image_height, self.image_width, 3)))

        # 卷积层1，步长为 1，relu 激活
        model.add(
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME"
            )
        )
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Activation("swish"))
        # 最大池化，池化窗口默认为 2
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
        # # 随机断连
        # model.add(tf.keras.layers.Dropout(rate=self.dropout_rate))

        # 卷积层2，步为 1，relu 激活
        model.add(
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
            )
        )
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Activation("swish"))
        # 最大池化，池化窗口默认为 2
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
        # # 随机断连
        # model.add(tf.keras.layers.Dropout(rate=self.dropout_rate))

        # 卷积层3，步为 1，relu 激活
        model.add(
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
            )
        )
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Activation("swish"))
        # 最大池化，池化窗口默认为 2
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
        # # 随机断连
        # model.add(tf.keras.layers.Dropout(rate=self.dropout_rate))

        # 卷积层3，步为 1，relu 激活
        model.add(
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
            )
        )
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Activation("swish"))
        # 最大池化，池化窗口默认为 2
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))
        # # 随机断连
        # model.add(tf.keras.layers.Dropout(rate=self.dropout_rate))

        # 平均池化
        # model.add(tf.keras.layers.AvgPool2D(pool_size=(3, 3)))

        # 需展平后才能与全连接层相连
        model.add(tf.keras.layers.Flatten())
        # 全连接层，输出为 1024，relu 激活
        model.add(tf.keras.layers.Dense(units=1024, activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=0.5))
        # 全连接层，Sigmoid 激活
        model.add(tf.keras.layers.Dense(units=self.max_captcha * self.char_set_len, activation="sigmoid"))
        return model

    # EfficientNet
    def model_efficientNet(self):
        # Pretrained backbone
        # efficientnetv2_b2_imagenet
        backbone = keras_cv.models.EfficientNetV2Backbone.from_preset('efficientnetv2_s_imagenet')
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(None, None, 3)),
                backbone,
                keras.layers.GlobalMaxPooling2D(),
                keras.layers.Dropout(rate=0.3),
                keras.layers.Dense(self.max_captcha * self.char_set_len, activation="sigmoid"),
            ]
        )
        return model
