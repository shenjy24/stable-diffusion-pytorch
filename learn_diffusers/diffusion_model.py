import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt


class BasicUNet(nn.Module):
    """A minimal UNet implementation."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU()  # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x


def corrupt(x, amount):
    """根据amount为输入x加入噪声，这就是退化过程"""
    noise = torch.rand_like(x)
    # 整理形状以保证广播机制不出错
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount


def draw():
    # 下载训练的数据集
    dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True,
                                         transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    x, y = next(iter(train_dataloader))
    print('Input shape:', x.shape)
    print('Labels:', y)
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

    # 绘制输入数据
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].set_title('Input data')
    axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

    # 加入噪声
    amount = torch.linspace(0, 1, x.shape[0])  # 从0到1 -> 退化更强烈了
    noised_x = corrupt(x, amount)

    # 绘制加噪版本的图像
    axs[1].set_title('Corrupted data (-- amount increases -->)')
    axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap='Greys')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    net = BasicUNet()
    x = torch.rand(8, 1, 28, 28)
    print(net(x).shape)