"""
MoE Layer Module
整合路由器和专家网络,实现完整的MoE前馈层
"""
import torch
from torch import nn
from typing import Optional, Tuple
from moe_router import MoERouter
from moe_experts import MoEExperts


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer
    
    结构:
    1. 路由器为每个token选择Top-K个专家
    2. 将token分配给对应的专家处理
    3. 使用门控权重加权组合专家输出
    
    特点:
    - 支持Top-K路由(通常K=2)
    - 负载均衡机制
    - 高效的批处理计算
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        use_aux_loss: bool = True,
        aux_loss_weight: float = 0.01,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            d_model: 隐藏层维度
            d_ff: FFN中间层维度
            n_experts: 专家数量
            top_k: 每个token激活的专家数
            use_aux_loss: 是否使用负载均衡损失
            aux_loss_weight: 辅助损失权重
        """
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        
        # 路由器
        self.router = MoERouter(
            d_model=d_model,
            n_experts=n_experts,
            top_k=top_k,
            use_aux_loss=use_aux_loss,
            aux_loss_weight=aux_loss_weight,
            device=device,
            dtype=dtype
        )
        
        # 专家网络
        self.experts = MoEExperts(
            n_experts=n_experts,
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )
        
        # 存储辅助损失
        self.aux_loss = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] 输入张量
        
        Returns:
            output: [batch_size, seq_len, d_model] MoE输出
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. 路由: 为每个token选择Top-K专家
        # gates: [batch_size, seq_len, top_k]
        # expert_indices: [batch_size, seq_len, top_k]
        gates, expert_indices, aux_loss = self.router(x)
        self.aux_loss = aux_loss
        
        # 2. 对每个Top-K位置分别处理
        output = torch.zeros_like(x)
        
        for k in range(self.top_k):
            # 获取第k个专家的索引和权重
            k_expert_idx = expert_indices[:, :, k]  # [batch_size, seq_len]
            k_gates = gates[:, :, k:k+1]  # [batch_size, seq_len, 1]
            
            # 通过专家网络
            k_output = self.experts(x, k_expert_idx)  # [batch_size, seq_len, d_model]
            
            # 加权累加
            output = output + k_gates * k_output
        
        return output
    
    def get_aux_loss(self) -> torch.Tensor:
        """获取辅助损失,用于训练时加到总损失中"""
        return self.aux_loss if self.aux_loss is not None else torch.tensor(0.0)


class EfficientMoELayer(nn.Module):
    """
    高效版MoE层
    
    优化:
    1. 批量处理所有专家的计算
    2. 使用scatter/gather操作减少循环
    3. 更适合大规模并行计算
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        use_aux_loss: bool = True,
        aux_loss_weight: float = 0.01,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        
        self.router = MoERouter(
            d_model=d_model,
            n_experts=n_experts,
            top_k=top_k,
            use_aux_loss=use_aux_loss,
            aux_loss_weight=aux_loss_weight,
            device=device,
            dtype=dtype
        )
        
        self.experts = MoEExperts(
            n_experts=n_experts,
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )
        
        self.aux_loss = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        高效前向传播实现
        
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        original_shape = x.shape
        batch_size, seq_len, d_model = original_shape
        
        # 路由
        gates, expert_indices, aux_loss = self.router(x)
        self.aux_loss = aux_loss
        
        # 展平batch和seq维度
        x_flat = x.reshape(-1, d_model)  # [batch_size * seq_len, d_model]
        gates_flat = gates.reshape(-1, self.top_k)  # [batch_size * seq_len, top_k]
        expert_indices_flat = expert_indices.reshape(-1, self.top_k)  # [batch_size * seq_len, top_k]
        
        # 初始化输出
        output_flat = torch.zeros_like(x_flat)
        
        # 为每个专家收集其需要处理的token
        for expert_id in range(self.n_experts):
            # 找到所有被路由到当前专家的位置
            expert_mask = (expert_indices_flat == expert_id)  # [batch_size * seq_len, top_k]
            
            if expert_mask.any():
                # 获取需要处理的token位置
                token_indices = expert_mask.any(dim=1).nonzero(as_tuple=True)[0]
                
                if len(token_indices) > 0:
                    # 提取这些token
                    expert_input = x_flat[token_indices]  # [n_tokens, d_model]
                    
                    # 通过专家处理
                    expert_output = self.experts.experts[expert_id](expert_input)
                    
                    # 计算每个token的权重
                    for i, token_idx in enumerate(token_indices):
                        # 找到当前专家在top_k中的位置
                        k_positions = (expert_indices_flat[token_idx] == expert_id).nonzero(as_tuple=True)[0]
                        for k_pos in k_positions:
                            weight = gates_flat[token_idx, k_pos]
                            output_flat[token_idx] += weight * expert_output[i]
        
        # 恢复形状
        output = output_flat.reshape(original_shape)
        return output
    
    def get_aux_loss(self) -> torch.Tensor:
        return self.aux_loss if self.aux_loss is not None else torch.tensor(0.0)
