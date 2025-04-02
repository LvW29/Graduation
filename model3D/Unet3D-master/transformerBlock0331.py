import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """图像分块嵌入模块，将输入图像转换为序列化的特征向量

    Args:
        in_channels (int): 输入图像的通道数
        patch_size (int): 图像分块的大小（正方形分块）
        embed_dim (int): 嵌入向量的维度
        num_patches (int): 图像分块的总数量（由原图尺寸和分块尺寸计算得出）
        dropout (float): Dropout层的丢弃概率
    """

    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()
        # 使用卷积层实现分块操作，卷积核和步长等于分块尺寸
        self.patcher = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)  # 保持batch维度，展平空间维度
        )

        # 可学习的分类标记（CLS token），初始化为随机值
        # self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        # 可学习的位置编码，包含num_patches+1个位置（分块数+CLS token）
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patches, embed_dim), requires_grad=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """前向传播过程

        Args:
            x (torch.Tensor): 输入图像张量，形状为 [B, C, H, W]

        Returns:
            torch.Tensor: 添加位置编码后的特征序列，形状为 [B, num_patches+1, embed_dim]
        """
        # 将CLS token扩展到当前batch的大小
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # 处理输入图像得到分块特征，调整维度为 [B, num_patches, embed_dim]
        x = self.patcher(x).permute(0, 2, 1)
        # 在特征序列前拼接CLS token
        # x = torch.cat([x, cls_token], dim=1)
        # x = x + cls_token
        # 添加位置编码并进行Dropout
        x = x + self.position_embedding

        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    """Transformer模型实现

    Args:
        in_channels (int): 输入图像的通道数
        patch_size (int): 图像分块的大小
        embed_dim (int): 嵌入向量维度
        num_patches (int): 分块总数量
        dropout (float): Dropout概率
        num_heads (int): 多头注意力机制的头数
        activation (str): 激活函数类型（如'relu'）
        num_encoders (int): Transformer编码器层数
        num_classes (int): 分类任务的类别数
    """

    def __init__(self, in_channels, embed_dim, dropout,
                 num_heads, activation, num_encoders):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        # 构建Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
                                                   activation=activation, batch_first=True, norm_first=True)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        # 分类头MLP，使用LayerNorm归一化
        # self.MLP = nn.Sequential(
        #     nn.LayerNorm(normalized_shape=embed_dim),
        #     nn.Linear(in_features=embed_dim, out_features=num_classes)
        # )
    def _reshape_output(self, x):
        # 调整维度顺序，从(N, C, D, H, W)变为(N, D, H, W, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        # 将特征展平为(N, seq_length, embedding_dim)
        x = x.view(x.size(0), -1, self.embed_dim)
        return x
    def forward(self, x):
        """前向传播过程

        Args:
            x (torch.Tensor): 输入图像张量，形状为 [B, C, D, H, W]

        Returns:
            torch.Tensor
        """
        x = self._reshape_output(x)
        # 通过Transformer编码器
        x = self.encoder_layers(x)
        # x = self.layer_norm(x)
        return x


def test():
    """测试函数，验证模型前向传播的维度转换是否正常"""
    # 生成随机输入数据（模拟batch_size=2的5D体积数据）
    x = torch.randn((2, 4, 32, 160, 160))
    # 初始化模型并进行预测
    model = Transformer(in_channels=4, embed_dim=512, dropout=0.1,
                num_heads=4, activation='relu', num_encoders=1)
    predict = model(x)
    # 输出形状验证
    print(x.shape)
    print(predict.shape)


if __name__ == "__main__":
    test()