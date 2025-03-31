import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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
    """改进的Transformer实现，参考Mamba的结构

    Args:
        in_channels (int): 输入通道数
        embed_dim (int): 嵌入维度
        num_heads (int): 注意力头数
        num_encoders (int): Transformer编码器层数
        dropout (float): Dropout率
    """

    def __init__(self, in_channels, embed_dim, num_heads, num_encoders, dropout=0.1):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoders = num_encoders

        # 输入投影层
        self.in_proj = nn.Linear(in_channels, embed_dim)

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 1600, embed_dim))  # 4*20*20=1600

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Layer Normalization
        self.norm = nn.LayerNorm(embed_dim)

    def _reshape_input(self, x):
        """将输入从(N, C, D, H, W)转换为(N, D*H*W, C)"""
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), -1, x.size(-1))
        return x

    def _reshape_output(self, x):
        """将输出从(N, D*H*W, C)转换回(N, C, D, H, W)"""
        x = x.view(x.size(0), 4, 20, 20, self.embed_dim)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x):
        # 输入维度转换
        x = self._reshape_input(x)

        # 输入投影
        x = self.in_proj(x)

        # 添加位置编码
        x = x + self.pos_embedding

        # Transformer编码器层
        x = self.encoder_layers(x)

        # Layer Normalization
        x = self.norm(x)

        # 输出投影
        x = self.out_proj(x)

        # 输出维度转换
        x = self._reshape_output(x)

        return x


def test():
    """测试函数"""
    # 生成随机输入数据
    x = torch.randn((2, 512, 4, 20, 20))
    # 初始化模型
    model = Transformer(
        in_channels=512,
        embed_dim=512,
        num_heads=8,
        num_encoders=6,
        dropout=0.1
    )
    # 前向传播
    predict = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {predict.shape}")


if __name__ == "__main__":
    test()
