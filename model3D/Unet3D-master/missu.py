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

class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()

        self.bn1 = nn.BatchNorm3d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        # 256, 128
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1



class SSU(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(SSU, self).__init__()

        self.num_classes = num_classes

        self.Softmax = nn.Softmax(dim=1)

        # 编码块和解码块
        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim//4, out_channels=self.embedding_dim//8)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim//8)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim//8, out_channels=self.embedding_dim//16)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim//16)

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim//16, out_channels=self.embedding_dim//32)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim//32)

        self.endconv = nn.Conv3d(self.embedding_dim // 32, 4, kernel_size=1)


    def decode(self, x1_1, x2_1, x3_1, x, intmd_x, intmd_layers=[1, 2, 3, 4]):

        assert intmd_layers is not None, "pass the intermediate layers for MLA"
        encoder_outputs = {}
        all_keys = []
        for i in intmd_layers:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()

        # 通过Enblock8_1和Enblock8_2进行编码
        x8 = encoder_outputs[all_keys[0]] # torch.Size([1, 4096, 512])
        x8 = self._reshape_output(x8) # torch.Size([1, 512, 16, 16, 16])
        x8 = self.Enblock8_1(x8) # torch.Size([1, 128, 16, 16, 16])
        x8 = self.Enblock8_2(x8) # torch.Size([1, 128, 16, 16, 16])

        # 通过DeUp4和DeBlock4进行解码
        y4 = self.DeUp4(x8, x3_1)  # (1, 64, 32, 32, 32)
        y4 = self.DeBlock4(y4) # torch.Size([1, 64, 32, 32, 32])

        # 通过DeUp3和DeBlock3进行解码
        y3 = self.DeUp3(y4, x2_1)  # (1, 32, 64, 64, 64)
        y3 = self.DeBlock3(y3)

        # 通过DeUp2和DeBlock2进行解码
        y2 = self.DeUp2(y3, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)

        # 通过endconv进行最终卷积，输出类别数为4
        y = self.endconv(y2)      # (1, 4, 128, 128, 128)
        y = self.Softmax(y)
        return y



class transUnetMSF(nn.Module):
    def __init__(self, args):
        super(transUnetMSF, self).__init__()

        self.vit = transformerBlock.Transformer(in_channels=512, embed_dim=512, dropout=0.1,
                num_heads=8, activation='relu', num_encoders=1)

        self.en1 = unet3dEncoder(4, 64)
        self.en2 = unet3dEncoder(64, 128)
        self.en3 = unet3dEncoder(128, 256)
        self.en4 = unet3dEncoder(256, 512)

        self.up3 = unet3dDecoder(512, 256)
        self.up2 = unet3dDecoder(256, 128)
        self.up1 = unet3dDecoder(128, 64)
        self.con_last = nn.Conv3d(64, 3, 1)

        # MSF
        self.Softmax = nn.Softmax(dim=1)
        self.embedding_dim = 512

        # 编码块和解码块
        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim)

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim, out_channels=self.embedding_dim//2)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim//2)

        # self.DeUp4 = DeUp_Cat(in_channels=256, out_channels=128)
        # self.DeBlock4 = DeBlock(in_channels=64)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim//2, out_channels=self.embedding_dim//4)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim//4)

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim//4, out_channels=self.embedding_dim//8)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim//8)

        self.endconv = nn.Conv3d(self.embedding_dim // 8, 3, kernel_size=1)

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

        # x = self.up3(x4, x3)
        # x = self.up2(x, x2)
        # x = self.up1(x, x1)
        # out = self.con_last(x)
        # decode过程
        # x4 = encoder_outputs[all_keys[0]]  # torch.Size([1, 4096, 512])
        # x4 = self._reshape_output(x4)  # torch.Size([1, 512, 16, 16, 16])

        # 初始 x4: torch.Size([2, 512, 4, 20, 20])  从transformer中出来的
        x4 = self.Enblock8_1(x4) # x4: torch.Size([2, 512, 4, 20, 20])
        x4 = self.Enblock8_2(x4)  # x4: torch.Size([2, 512, 4, 20, 20])

        # 通过DeUp4和DeBlock4进行解码
        # 初始 x3: torch.Size([2, 256, 8, 40, 40])
        y4 = self.DeUp4(x4, x3)  # (1, 64, 32, 32, 32)
        y4 = self.DeBlock4(y4)  # torch.Size([1, 64, 32, 32, 32])

        # 通过DeUp3和DeBlock3进行解码
        y3 = self.DeUp3(y4, x2)  # (1, 32, 64, 64, 64)
        y3 = self.DeBlock3(y3)

        # 通过DeUp2和DeBlock2进行解码
        y2 = self.DeUp2(y3, x1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2)

        # 通过endconv进行最终卷积，输出类别数为4
        y = self.endconv(y2)  # (1, 4, 128, 128, 128)
        y = self.Softmax(y)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
