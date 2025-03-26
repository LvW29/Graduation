import torch.nn as nn
import torch.nn.functional as F
import torch
import MFEblock2
import MFEblock
import transformerBlock
# adapt from https://github.com/MIC-DKFZ/BraTS2017

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

def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m



class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout)

        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y



class UnetTransMFE2(nn.Module):
    def __init__(self, args):
        super(UnetTransMFE2, self).__init__()
        self.in_channels = 4
        self.base_channels = 64

        self.InitConv = InitConv(in_channels=self.in_channels, out_channels=self.base_channels, dropout=0.2)
        self.EnBlock1 = EnBlock(in_channels=self.base_channels)
        self.EnDown1 = EnDown(in_channels=self.base_channels, out_channels=self.base_channels*2)

        self.EnBlock2_1 = EnBlock(in_channels=self.base_channels*2)
        self.EnBlock2_2 = EnBlock(in_channels=self.base_channels*2)
        self.EnDown2 = EnDown(in_channels=self.base_channels*2, out_channels=self.base_channels*4)

        self.EnBlock3_1 = EnBlock(in_channels=self.base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=self.base_channels * 4)
        self.EnDown3 = EnDown(in_channels=self.base_channels*4, out_channels=self.base_channels*8)

        self.EnBlock4_1 = EnBlock(in_channels=self.base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=self.base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=self.base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=self.base_channels * 8)

        self.MFE = MFEblock.MFEblock(in_channels=self.base_channels, atrous_rates=[2, 4, 6])
        self.MFE1 = MFEblock2.MFEblock(in_channels=64, atrous_rates=[2, 4, 8])
        self.MFE2 = MFEblock2.MFEblock(in_channels=128, atrous_rates=[2, 4, 6])
        self.MFE3 = MFEblock2.MFEblock(in_channels=256, atrous_rates=[2, 3, 4])
        # self.MFE4 = MFEblock.MFEblock(in_channels=64, atrous_rates=[2, 3, 4])
        self.up3 = unet3dDecoder(512, 256)
        self.up2 = unet3dDecoder(256, 128)
        self.up1 = unet3dDecoder(128, 64)
        self.con_last = nn.Conv3d(64, 3, 1)

        self.vit = transformerBlock.Transformer(in_channels=512, embed_dim=512, dropout=0.1,
                num_heads=8, activation='relu', num_encoders=6)

    def _reshape_output(self, x):
        # 将特征从(N, seq_length, embedding_dim)变为(N, D, H, W, embedding_dim)
        x = x.view(
            x.size(0),
            4,
            20,
            20,
            512,
        )
        # 调整维度顺序，从(N, D, H, W, embedding_dim)变为(N, embedding_dim, D, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x

    def forward(self, x):
        x = self.InitConv(x) # torch.Size([1, 64, 32, 160, 160])
        x1_1 = self.MFE(x)
        # x1_1 = self.MFE1(x) # torch.Size([1, 64, 32, 160, 160])
        # x1_2 = self.EnDown1(x1_1)  # torch.Size([1, 128, 16, 80, 80])

        x2_1 = self.MFE1(x1_1) # torch.Size([1, 128, 16, 80, 80])
        # x2_1 = self.EnBlock2_2(x2_1) # torch.Size([1, 128, 16, 80, 80])
        # x2_2 = self.EnDown2(x2_1)  # torch.Size([1, 256, 8, 40, 40])

        x3_1 = self.MFE2(x2_1) # torch.Size([1, 256, 8, 40, 40])
        # x3_1 = self.EnBlock3_2(x3_1) # torch.Size([1, 256, 8, 40, 40])
        # x3_2 = self.EnDown3(x3_1)  # torch.Size([1, 512, 4, 20, 20])

        x4_1 = self.MFE3(x3_1) # torch.Size([1, 512, 4, 20, 20])
        x4_2 = self.EnBlock4_2(x4_1) # torch.Size([1, 512, 4, 20, 20])
        x4_3 = self.EnBlock4_3(x4_2) # torch.Size([1, 512, 4, 20, 20])
        output = self.EnBlock4_4(x4_3)  # torch.Size([1, 512, 4, 20, 20])

        output = self.vit(output) # x4.shape: torch.Size([2, 1600, 512])
        output = self._reshape_output(output)  # x4.shape: torch.Size([2, 512, 4, 20, 20])

        x = self.up3(output, x3_1)
        x = self.up2(x, x2_1)
        x = self.up1(x, x1_1)
        out = self.con_last(x)
        return out


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.rand((1, 4, 32, 160, 160), device=device)
        # model = Unet1(in_channels=4, base_channels=16, num_classes=4)
        model = UnetTransMFE2(args="")
        model.to(device)
        output = model(x)
        print('output:', output.shape)