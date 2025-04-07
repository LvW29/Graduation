import torch
import torch.nn as nn
from mamba_ssm_self.modules.mamba_simple import Mamba


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
        self.pos_embedding = nn.Parameter(torch.randn(1, 8000, embed_dim))  # 4*20*20=1600

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

    def forward(self, x):
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

        return x


class testTime(nn.Module):
    def __init__(self, args):
        super(testTime, self).__init__()

        self.mamba = Mamba(
            d_model=512,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2)  # Block expansion factor)

        self.trans = Transformer(
            in_channels=512,
            embed_dim=512,
            num_heads=8,
            num_encoders=1,
            dropout=0.1
        )

    def forward(self, x):
        # 计时逻辑开始
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        # x = self.mamba(x)  # 被测模块
        x = self.trans(x)  # 被测模块

        end_event.record()
        torch.cuda.synchronize()
        # 计时逻辑结束

        # 显存统计
        # mem_used = torch.cuda.max_memory_allocated() / (1024**2)  # 转换为MB


if __name__ == '__main__':
    with torch.no_grad():
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 测试配置参数
        warmup_iters = 50  # 预热迭代次数
        test_iters = 1000  # 正式测试迭代次数
        input_shape = (1, 8000, 512)  # 输入张量形状

        # 初始化模型
        model = testTime(args="")
        model.to(device)
        model.eval()  # 设置为评估模式

        # 创建计时事件
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # 预热阶段（不记录时间）
        print("Warming up...")
        for _ in range(warmup_iters):
            dummy_input = torch.rand(input_shape, device=device)
            _ = model(dummy_input)

        # 正式测量阶段
        print("Benchmarking...")
        timings = []
        # 在测试循环中添加显存统计
        mem_stats = []
        torch.cuda.nvtx.range_push("frequency_lock")
        for _ in range(test_iters):
            # 每次使用新生成的数据（更接近真实场景）
            x = torch.rand(input_shape, device=device)

            # torch.cuda.empty_cache()
            # torch.cuda.reset_peak_memory_stats()

            torch.cuda.synchronize()
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()

            timings.append(starter.elapsed_time(ender))
            mem_stats.append(torch.cuda.max_memory_allocated() / (1024 ** 2))

        # 统计结果
        avg_time = sum(timings) / test_iters
        std_time = torch.std(torch.tensor(timings)).item()
        print(f"\nBenchmark results (n={test_iters}):")
        print(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"Max time: {max(timings):.2f} ms")
        print(f"Min time: {min(timings):.2f} ms")

        # 显存统计输出
        # avg_mem = sum(mem_stats) / test_iters
        # print(f"\nMemory benchmark (n={test_iters}):")
        # print(f"Average memory usage: {avg_mem:.2f} ± {torch.std(torch.tensor(mem_stats)):.2f} MB")
        # print(f"Max memory: {max(mem_stats):.2f} MB")
        # print(f"Min memory: {min(mem_stats):.2f} MB")

        # # 验证输出形状
        # test_input = torch.rand(input_shape, device=device)
        # output = model(test_input)
        # print("\nOutput shape verification:", output.shape)