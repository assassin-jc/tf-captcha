## 图片文件夹
```
origin_image_dir = "./sample/origin/"  # 原始文件
```
## 模型文件夹
```
model_save_dir = "./model/"  # 训练好的模型储存路径
```
## 图片相关参数
```
image_width = 80  # 图片宽度
image_height = 40  # 图片高度
max_captcha = 4  # 验证码字符个数
image_suffix = "jpg"  # 图片文件后缀
```

## 验证码字符相关参数
```
char_set = "0123456789abcdefghijklmnopqrstuvwxyz"
char_set = "abcdefghijklmnopqrstuvwxyz"
char_set = "0123456789"
```
## 训练相关参数
```
epochs = 10  # 到指定世代后停止
enable_gpu = 0  # 使用GPU还是CPU,使用GPU需要安装对应版本的tensorflow-gpu==1.7.0
dropout_rate = 0.25
```