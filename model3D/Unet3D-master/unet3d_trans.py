import torch
import torch.nn as nn
import transformerBlock


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


class unet3d_trans(nn.Module):
    def __init__(self, args):
        super(unet3d_trans, self).__init__()

        self.vit = transformerBlock.Transformer(in_channels=512, embed_dim=512, dropout=0.1,
                num_heads=8, activation='relu', num_encoders=6)

        self.en1 = unet3dEncoder(4, 64)
        self.en2 = unet3dEncoder(64, 128)
        self.en3 = unet3dEncoder(128, 256)
        self.en4 = unet3dEncoder(256, 512)

        self.up3 = unet3dDecoder(512, 256)
        self.up2 = unet3dDecoder(256, 128)
        self.up1 = unet3dDecoder(128, 64)
        self.con_last = nn.Conv3d(64, 3, 1)

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
        x1, x = self.en1(x)
        x2, x = self.en2(x)
        x3, x = self.en3(x)
        x4, _ = self.en4(x) # x4.shape: torch.Size([2, 512, 4, 20, 20])

        x4 = self.vit(x4) # x4.shape: torch.Size([2, 1600, 512])
        x4 = self._reshape_output(x4)  # x4.shape: torch.Size([2, 512, 4, 20, 20])

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
