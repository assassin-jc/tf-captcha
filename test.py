import json

import keras
import numpy as np
import tensorflow as tf
from PIL import Image

from train_model import char_accuracy


def main(img_path):
    with open("conf/sample_config.json", "r") as f:
        sample_conf = json.load(f)
    char_set = sample_conf["char_set"]
    captcha_image = Image.open(img_path)
    # 将图片转换为数组
    array = tf.keras.utils.img_to_array(captcha_image)
    # 转为黑白
    data = tf.image.rgb_to_grayscale(array) / 255.0
    data = np.reshape(data, (1, 60, 120, 1))

    # model = tf.keras.models.load_model("model/model.keras" )
    model = keras.models.load_model(
        "model/model.keras",
        custom_objects={"char_accuracy": char_accuracy},
    )
    predit = model.predict(data)
    max_id_p = tf.argmax(tf.reshape(predit, (4, 62)), 1)

    recongized = ""
    for id in max_id_p:
        recongized += char_set[id]

    print(recongized)


if __name__ == '__main__':
    main("sample/origin/0TFh_17180692688244812.png")
