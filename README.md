

本项目针对字符型图片验证码，使用tensorflow2.0版本实现卷积神经网络，进行验证码识别。
项目代码主要参照 https://github.com/nickliqian/cnn_captcha, 主要是使用tensorflow2.0 重新实现相关功能也顺带学习神经网络

实际尝试下来,自建模型的训练速度和准确率都不如efficientnetv2(keras的预训练模型)

![自建模型字符识别准确率曲线](https://github.com/assassin-jc/tf-captcha/assets/9713245/2e9f760f-0c38-4636-a310-6b8e447830b7 "自建模型字符识别准确率曲线")

![efficientnetv2字符识别准确率曲线](https://github.com/assassin-jc/tf-captcha/assets/9713245/624dcb37-c44f-4033-98a0-73901849eb2a "efficientnetv2字符识别准确率曲线")
