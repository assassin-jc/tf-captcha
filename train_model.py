# -*- coding: utf-8 -*-
import json
import os
import random
import time

import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from cnnlib.network import CNN


class TrainError(Exception):
    pass


@keras.saving.register_keras_serializable()
class char_accuracy(keras.metrics.Metric):
    def __init__(self, batch_size, char_length, charset_length, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.char_length = char_length
        self.charset_length = charset_length
        self.total = self.add_weight(name='total',
                                     dtype=tf.int32,
                                     initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count',
                                     dtype=tf.int32,
                                     initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        max_id_p = tf.argmax(tf.reshape(y_pred, (self.batch_size, self.char_length, self.charset_length)), 2)
        max_id_l = tf.argmax(tf.reshape(y_true, (self.batch_size, self.char_length, self.charset_length)), 2)
        correct = tf.reduce_sum(tf.cast(tf.equal(max_id_p, max_id_l), tf.float32))
        self.count.assign_add(correct)
        self.total.assign_add(self.batch_size * self.char_length)

    def result(self):
        return self.count / self.total

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "batch_size": self.batch_size,
                "char_length": self.char_length,
                "charset_length": self.charset_length,
                # "name": self.name,
            }
        )
        return config


class TrainModel(CNN):
    def __init__(self, img_path, char_set, model_save_dir, epochs, image_suffix, batch_size, dropout_rate,
                 verify=False):
        # 训练相关参数
        self.epochs = epochs
        self.batch_size = batch_size

        self.image_suffix = image_suffix
        char_set = [str(i) for i in char_set]

        # 打乱文件顺序+校验图片格式
        self.img_path = img_path
        self.images_list = os.listdir(img_path)
        # 校验格式
        if verify:
            self.confirm_image_suffix()

        # 获得图片宽高和字符长度基本信息
        label, captcha_array = self.gen_captcha_text_image(img_path, self.images_list[0])

        captcha_shape = captcha_array.shape
        captcha_shape_len = len(captcha_shape)
        if captcha_shape_len == 3:
            image_height, image_width, channel = captcha_shape
            self.channel = channel
        elif captcha_shape_len == 2:
            image_height, image_width = captcha_shape
        else:
            raise TrainError("图片转换为矩阵时出错，请检查图片格式")

        # 初始化变量
        super(TrainModel, self).__init__(image_height, image_width, len(label), char_set, model_save_dir, dropout_rate)

        # 相关信息打印
        print("-->图片尺寸: {} X {}".format(image_height, image_width))
        print("-->验证码长度: {}".format(self.max_captcha))
        print("-->验证码共{}类 {}".format(self.char_set_len, char_set))

    @staticmethod
    def gen_captcha_text_image(img_path, img_name):
        """
        返回一个验证码的array形式和对应的字符串标签
        :return:tuple (str, numpy.array)
        """

        # 标签
        label = img_name.split("_")[0]
        # 文件
        img_file = os.path.join(img_path, img_name)
        captcha_image = Image.open(img_file)
        captcha_array = tf.keras.utils.img_to_array(captcha_image)  # 向量化
        return label, captcha_array

    def get_data(self):
        size = len(self.images_list)
        batch_x = np.zeros([size, self.image_height, self.image_width, 1])  # 初始化
        batch_y = np.zeros([size, self.max_captcha * self.char_set_len])  # 初始化

        for i, img_name in enumerate(self.images_list):
            label, image_array = self.gen_captcha_text_image(self.img_path, img_name)
            image_array = tf.image.rgb_to_grayscale(image_array)  # 灰度化图片
            data = image_array / 255.0  # 归一化
            batch_x[i, :] = tf.reshape(data, (self.image_height, self.image_width, 1))
            batch_y[i, :] = self.text2vec(label)  # 生成 oneHot

        # 分割训练集/测试集
        x_train, x_val, y_train, y_val = train_test_split(batch_x, batch_y, test_size=0.2)

        return x_train, x_val, y_train, y_val

    def confirm_image_suffix(self):
        # 在训练前校验所有文件格式
        print("开始校验所有图片后缀")
        for index, img_name in enumerate(self.train_images_list):
            print("{} image pass".format(index), end='\r')
            if not img_name.endswith(self.image_suffix):
                raise TrainError('confirm images suffix：you request [.{}] file but get file [{}]'
                                 .format(self.image_suffix, img_name))
        print("所有图片格式校验通过")

    def train_cnn(self):
        # 数据处理
        x_train, x_val, y_train, y_val = self.get_data()

        train_model = self.model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # 编译模型，Adam 优化器，多分类交叉熵损失函数，准确度评估
        train_model.compile(optimizer=optimizer, loss="binary_crossentropy",
                            metrics=[char_accuracy(self.batch_size, self.max_captcha, self.char_set_len)])
        # 模型保存对象
        checkpoint = ModelCheckpoint(filepath=self.model_save_dir + 'model.keras', verbose=1, save_best_only=True)

        # 模型训练及评估
        train_model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                        validation_data=(x_val, y_val),
                        callbacks=[checkpoint])


def main():
    keras.saving.get_custom_objects().clear()
    with open("conf/sample_config.json", "r") as f:
        sample_conf = json.load(f)

    image_dir = sample_conf["image_dir"]
    model_save_dir = sample_conf["model_save_dir"]
    epochs = sample_conf["epochs"]
    enable_gpu = sample_conf["enable_gpu"]
    image_suffix = sample_conf['image_suffix']
    batch_size = sample_conf['batch_size']
    dropout_rate = sample_conf['dropout_rate']
    char_set = sample_conf["char_set"]

    if not enable_gpu:
        # 设置以下环境变量可开启CPU识别
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    tm = TrainModel(image_dir, char_set, model_save_dir, epochs, image_suffix, batch_size, dropout_rate, verify=False)
    tm.train_cnn()  # 开始训练模型


if __name__ == '__main__':
    main()
