import os

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image

char_set = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def text2vec(text):
    """
    转标签为oneHot编码
    :param text: str
    :return: numpy.array
    """
    text_len = len(text)

    vector = torch.zeros(4 * len(char_set))

    for i, ch in enumerate(text):
        idx = i * len(char_set) + char_set.index(ch)
        vector[idx] = 1
    return vector


class CaptchaDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform

        # 加载图像文件名和标签
        self.images = []
        self.labels = []
        for filename in os.listdir(root):
            # 将图像文件名转换为所需的二维张量标签
            # 标签
            label = filename.split("_")[0]
            self.images.append(os.path.join(root, filename))
            self.labels.append(text2vec(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[index]


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


def data_prepossessing():
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        # converting to tensor
        transforms.ToTensor(),
        # ResNet normalization
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # images_list = os.listdir("sample/origin")
    image_dataset = CaptchaDataset("sample/origin", data_transforms)

    train_set, validate_set = torch.utils.data.random_split(image_dataset, [16000, 4000])

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    validate_dataloader = torch.utils.data.DataLoader(validate_set, batch_size=8, shuffle=False)
    return train_dataloader, validate_dataloader


def train_cnn():
    train_loader, test_loader = data_prepossessing()
    model = nn.Sequential(
        nn.Conv2d(1, 32, (3, 3), 1, "same"),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),  # 激活函数类
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.25),
        nn.Conv2d(32, 64, (3, 3), 1, "same"),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),  # 激活函数类
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.25),
        nn.Conv2d(64, 128, (3, 3), 1, "same"),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),  # 激活函数类
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.25),
        nn.Conv2d(128, 256, (3, 3), 1, "same"),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),  # 激活函数类
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.25),
        Flatten(),
        nn.Linear(3 * 7 * 256, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 62 * 4),
        nn.Sigmoid()
    )

    if os.path.exists("model/model.pth"):
        checkpoint = torch.load("model/model.pth", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

    fit(model, train_loader, test_loader, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=0.0005), 10)
    torch.save(model.state_dict(), "model/model.pth")


def fit(model, train_dataloader, val_dataloader, loss_fn, optimizer, epochs):
    print("Start training, please be patient.")
    # 全数据集迭代 epochs 次
    for epoch in range(epochs):
        # 从数据加载器中读取 Batch 数据开始训练
        for i, (images, labels) in enumerate(train_dataloader):
            # labels = labels  # 真实标签
            outputs = model(images)  # 前向传播
            loss = loss_fn(outputs, labels)  # 传入模型输出和真实标签
            optimizer.zero_grad()  # 优化器梯度清零，否则会累计
            loss.backward()  # 从最后 loss 开始反向传播
            optimizer.step()  # 优化器迭代
            # 自定义训练输出样式
            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Batch [{}/{}], Train loss: {:.3f}".format(
                        epoch + 1, epochs, i + 1, len(train_dataloader), loss.item()
                    )
                )
        # 每个 Epoch 执行一次测试
        correct = 0
        total = 0
        for images, labels in val_dataloader:
            outputs = model(images)
            # 得到输出最大值 _ 及其索引 predicted
            max_id_p = torch.argmax(torch.reshape(outputs.data, (8, 4, 62)), 2)
            max_id_l = torch.argmax(torch.reshape(labels, (8, 4, 62)), 2)
            correct += (max_id_p == max_id_l).sum().item()  # 如果预测结果和真实值相等则计数 +1
            total += labels.size(0) * 4  # 总测试样本数据计数
        print(
            "============ Test accuracy: {:.3f} =============".format(correct / total)
        )


if __name__ == '__main__':
    train_cnn()
