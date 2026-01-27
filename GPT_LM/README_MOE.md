# 企业级 MoE Transformer 实现

这是一个参考 DeepSeek V2 设计理念的 Mixture of Experts (MoE) Transformer 实现。

## 🚀 核心特性

### 1. **完整的 MoE 实现**
- Top-K 路由机制 (通常 K=2)
- 负载均衡辅助损失
- 多专家并行计算
- 门控网络 (Gating Network)

### 2. **灵活的架构选择**
- **全MoE模型**: 所有层都使用 MoE
- **混合模型**: 部分层用 MoE,部分层用标准 FFN
- **可配置专家数**: 支持 4/8/16/32/64 等任意专家数量

### 3. **训练优化**
- 负载均衡损失防止专家崩塌
- 门控权重归一化
- 支持梯度检查点 (可选)
- 辅助损失自动聚合

### 4. **生产就绪**
- 类型提示完整
- 详细的文档注释
- 模块化设计,易于扩展
- 完整的使用示例

### MoE Block 结构

```
输入 x [batch, seq, d_model]
    ↓
LayerNorm
    ↓
Multi-Head Attention
    ↓
残差连接 (+)
    ↓
LayerNorm
    ↓
┌─────────── MoE Layer ───────────┐
│                                  │
│  Router (门控网络)               │
│    ↓                             │
│  Top-K Selection                 │
│    ↓                             │
│  ┌──────┬──────┬─────┬──────┐  │
│  │Expert│Expert│ ... │Expert│  │
│  │  1   │  2   │     │  N   │  │
│  └──────┴──────┴─────┴──────┘  │
│    ↓                             │
│  加权组合 (按门控权重)           │
│                                  │
└──────────────────────────────────┘
    ↓
残差连接 (+)
    ↓
输出
```

### 关键组件

1. **MoERouter** (`moe_router.py`)
   - 计算每个 token 对所有专家的亲和度
   - 选择 Top-K 个最相关的专家
   - 计算门控权重
   - 负载均衡损失

2. **MoEExperts** (`moe_experts.py`)
   - 包含 N 个独立的专家网络
   - 每个专家是一个 SwiGLU FFN
   - 并行处理分配的 tokens

3. **MoELayer** (`moe_layer.py`)
   - 整合 Router 和 Experts
   - 处理 token 到专家的分发
   - 加权组合专家输出

4. **MoETransformerBlock** (`moe_transformer_block.py`)
   - 标准 Attention + MoE FFN
   - Pre-norm 架构
   - RMSNorm 归一化

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `moe_router.py` | 门控路由器,负责 token-expert 匹配 |
| `moe_experts.py` | 专家网络集合 |
| `moe_layer.py` | 完整的 MoE 层实现 |
| `moe_transformer_block.py` | MoE Transformer Block |
| `moe_transformer.py` | 完整的 MoE 语言模型 |
| `moe_examples.py` | 完整使用示例 |

## 🎯 快速开始

### 1. 基础 MoE 模型

```python
from moe_transformer import MoETransformerLM

# 创建模型
model = MoETransformerLM(
    d_model=512,
    n_head=8,
    vocab_size=10000,
    max_seq_len=1024,
    d_ff=2048,
    theta=10000.0,
    n_layer=12,
    n_experts=8,      # 每层8个专家
    top_k=2,          # 每个token激活2个专家
    use_moe_aux_loss=True,
    moe_aux_loss_weight=0.01,
)

# 前向传播
import torch
token_ids = torch.randint(0, 10000, (4, 128))  # [batch, seq]
logits = model(token_ids)  # [batch, seq, vocab]
```

### 2. 混合架构模型

```python
from moe_transformer import HybridMoETransformerLM

# 仅在指定层使用 MoE
model = HybridMoETransformerLM(
    d_model=512,
    n_head=8,
    vocab_size=10000,
    max_seq_len=1024,
    d_ff=2048,
    theta=10000.0,
    n_layer=12,
    moe_layer_indices=[2, 5, 8, 11],  # 仅这4层用MoE
    n_experts=8,
    top_k=2,
)
```

### 3. 训练循环

```python
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    # 前向传播
    logits = model(batch['input_ids'])
    
    # 语言模型损失
    lm_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        batch['labels'].view(-1)
    )
    
    # MoE 辅助损失
    aux_loss = model.get_aux_loss()
    
    # 总损失
    total_loss = lm_loss + aux_loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

## 📖 详细说明

### MoE 核心概念

#### 2. Top-K 路由

```
例如: 8 个专家,Top-2 路由

Token: "artificial"
    ↓
Router 计算得分: [0.3, 0.1, 0.05, 0.25, 0.08, 0.12, 0.05, 0.05]
    ↓
选择 Top-2: Expert 1 (0.3), Expert 4 (0.25)
    ↓
归一化权重: [0.545, 0.455]
    ↓
输出 = 0.545 * Expert1(token) + 0.455 * Expert4(token)
```

#### 3. 负载均衡

**问题**: 训练时某些专家可能被过度使用,其他专家欠使用

**解决方案**: 辅助损失惩罚不均匀分布

```python
# 计算每个专家的使用频率
expert_usage = [0.4, 0.3, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]

# 理想情况: 每个专家 1/8 = 0.125
# 辅助损失惩罚偏离理想值的情况
```

### 配置建议

| 模型规模 | n_experts | top_k | d_ff | 说明 |
|---------|-----------|-------|------|------|
| 小型 | 4 | 2 | 1024 | 快速实验 |
| 中型 | 8 | 2 | 2048 | 平衡性能 |
| 大型 | 16 | 2 | 4096 | 高性能 |
| 超大型 | 64 | 2-4 | 8192 | DeepSeek级别 |

### 超参数调优

```python
# 辅助损失权重
# - 太小: 负载不均衡,专家崩塌
# - 太大: 影响主任务性能
moe_aux_loss_weight = 0.01  # 推荐范围: 0.001 - 0.1

# Top-K 选择
# - K=1: 最稀疏,但可能容量不足
# - K=2: 平衡选择 (推荐)
# - K>2: 更多容量,但计算增加
top_k = 2
```

## ⚡ 性能优化

### 1. 内存优化

```python
# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(input_ids)
    loss = compute_loss(logits, labels) + model.get_aux_loss()

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. 计算优化

```python
# 减少专家数量但增加容量
# 方案A: 8专家 × 2048 FFN = 16,384 参数单元
# 方案B: 4专家 × 4096 FFN = 16,384 参数单元
# 方案B计算更快但专业化程度低
```

### 3. 混合架构策略

```python
# DeepSeek V2 风格: 浅层用标准FFN,深层用MoE
n_layer = 32
moe_layers = list(range(16, 32))  # 后半部分用MoE

model = HybridMoETransformerLM(
    n_layer=n_layer,
    moe_layer_indices=moe_layers,
    ...
)
```


## 🧪 实验建议

### 1. 消融实验

```python
# 测试专家数量的影响
for n_experts in [4, 8, 16, 32]:
    model = MoETransformerLM(n_experts=n_experts, ...)
    # 训练和评估
```

### 2. Top-K 实验

```python
# 测试激活专家数的影响
for top_k in [1, 2, 4]:
    model = MoETransformerLM(top_k=top_k, ...)
    # 对比性能和计算成本
```

### 3. 混合策略

```python
# 测试不同的MoE层分布
strategies = {
    "all": list(range(12)),          # 所有层
    "deep": list(range(6, 12)),      # 深层
    "sparse": [2, 5, 8, 11],         # 稀疏分布
}
```

## 📝 最佳实践

1. **从小规模开始**: 先用 4-8 个专家验证想法
2. **监控负载**: 确保专家被均匀使用
3. **调整辅助损失**: 根据任务调整权重
4. **使用混合架构**: 不是所有层都需要 MoE
5. **梯度裁剪**: MoE 训练可能不稳定,使用梯度裁剪

## 🔗 参考资料

- [Shazeer et al. 2017 - Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)

## 📄 许可

本实现仅供学习和研究使用。

---

**作者**: Which_W 

