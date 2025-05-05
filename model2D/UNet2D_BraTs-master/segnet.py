import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(x)))
        x, indices = nn.MaxPool2d(2, 2, return_indices=True)(y)
        return x, y, indices


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, indices, output_size):
        x = self.unpool(x, indices, output_size=output_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class segnet(nn.Module):
    def __init__(self, args):
        super(segnet, self).__init__()
        in_chan = 4
        out_chan = 3
        # 编码器
        self.down1 = Downsample_block(in_chan, 64)
        self.down2 = Downsample_block(64, 128)
        self.down3 = Downsample_block(128, 256)
        self.down4 = Downsample_block(256, 512)
        self.down5 = Downsample_block(512, 512)
        # 瓶颈
        self.conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        # 解码器
        self.up5 = Upsample_block(512, 512)
        self.up4 = Upsample_block(512, 256)
        self.up3 = Upsample_block(256, 128)
        self.up2 = Upsample_block(128, 64)
        self.up1 = Upsample_block(64, 64)
        self.outconv = nn.Conv2d(64, out_chan, 1)

    def forward(self, x):
        # 编码器
        x, y1, indices1 = self.down1(x)
        size1 = y1.size()
        x, y2, indices2 = self.down2(x)
        size2 = y2.size()
        x, y3, indices3 = self.down3(x)
        size3 = y3.size()
        x, y4, indices4 = self.down4(x)
        size4 = y4.size()
        x, y5, indices5 = self.down5(x)
        size5 = y5.size()

        # 瓶颈
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # 解码器
        x = self.up5(x, indices5, output_size=size5)
        x = self.up4(x, indices4, output_size=size4)
        x = self.up3(x, indices3, output_size=size3)
        x = self.up2(x, indices2, output_size=size2)
        x = self.up1(x, indices1, output_size=size1)
        x = self.outconv(x)

        return x


if __name__ == '__main__':
    with torch.no_grad():
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import torch

        # 模拟数据
        x = torch.randn(2, 4, 160, 160)  # batch_size=2, 4通道, 160×160
        model = segnet(args="")
        output = model(x)
        print(output.shape)  # 输出: torch.Size([2, 10, 160, 160])