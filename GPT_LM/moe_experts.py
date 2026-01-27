"""
MoE Experts Module
实现多个独立的FFN专家网络
"""
import torch
from torch import nn
from typing import Optional
from swiGLU import SwiGLU


class Expert(nn.Module):
    """单个专家网络，使用SwiGLU激活函数"""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MoEExperts(nn.Module):
    """
    多专家系统
    包含n_experts个独立的专家网络
    """
    
    def __init__(
        self,
        n_experts: int,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        
        # 创建多个专家
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, device=device, dtype=dtype)
            for _ in range(n_experts)
        ])
    
    def forward(self, x: torch.Tensor, expert_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] 输入张量
            expert_idx: [batch_size, seq_len] 每个token分配的专家索引
        
        Returns:
            output: [batch_size, seq_len, d_model] 专家输出
        """
        batch_size, seq_len, d_model = x.shape
        
        # 重塑为 [batch_size * seq_len, d_model]
        x_flat = x.reshape(-1, d_model)
        expert_idx_flat = expert_idx.reshape(-1)
        
        # 初始化输出
        output = torch.zeros_like(x_flat)
        
        # 对每个专家分别处理
        for expert_id in range(self.n_experts):
            # 找到分配给当前专家的所有token
            expert_mask = (expert_idx_flat == expert_id)
            if expert_mask.any():
                # 提取这些token
                expert_input = x_flat[expert_mask]
                # 通过专家处理
                expert_output = self.experts[expert_id](expert_input)
                # 写回输出
                output[expert_mask] = expert_output
        
        # 恢复形状
        output = output.reshape(batch_size, seq_len, d_model)
        return output
