import torch
import torch.nn as nn


class Double3DConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Double3DConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class unet3dEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(unet3dEncoder, self).__init__()
        self.conv = Double3DConv(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)


class unet3dDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unet3dDecoder, self).__init__()
        self.sample = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.conv = Double3DConv(in_channels, out_channels)
    def forward(self, x, x1):
        x = self.sample(x)
        x = torch.cat((x, x1), dim=1)
        x = self.conv(x)
        return x


class unet3d(nn.Module):
    def __init__(self, args):
        super(unet3d, self).__init__()
        init_channels = 4
        class_nums = 3

        self.en1 = unet3dEncoder(init_channels, 64)
        self.en2 = unet3dEncoder(64, 128)
        self.en3 = unet3dEncoder(128, 256)
        self.en4 = unet3dEncoder(256, 512)

        self.up3 = unet3dDecoder(512, 256)
        self.up2 = unet3dDecoder(256, 128)
        self.up1 = unet3dDecoder(128, 64)
        self.con_last = nn.Conv3d(64, class_nums, 1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x = self.en1(x)
        x2, x = self.en2(x)
        x3, x = self.en3(x)
        x4, _ = self.en4(x)

        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.con_last(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.rand((1, 4, 32, 160, 160), device=device)
        # model = Unet1(in_channels=4, base_channels=16, num_classes=4)
        model = unet3d(args="")
        model.to(device)
        output = model(x)
        print('output:', output.shape)