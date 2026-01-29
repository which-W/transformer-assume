# Transformer 深度学习项目

> ⚠️ **重要声明：本项目仅用于学习目的，请勿用于任何商业用途！**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-教育学习用途-red.svg)](LICENSE)

## 📚 项目简介

这是一个个人深度学习学习项目，专注于从零开始实现 Transformer 架构及其变体。**项目完全出于学习和研究目的开发，不具备商业应用价值，严禁用于任何商业场景。**

### 🎯 学习目标

- 深入理解 Transformer 架构的核心原理
- 掌握多头注意力机制的实现细节
- 学习从底层构建深度学习模型
- 探索 Mixture of Experts (MoE) 等前沿技术
- 实践 PyTorch 深度学习框架的使用

### ⚠️ 使用限制

1. **仅限学习研究**：本项目仅用于个人学习和教育目的
2. **禁止商业使用**：严禁用于任何商业产品或服务
3. **不保证性能**：模型性能未经过严格验证
4. **无技术支持**：不提供任何形式的技术支持

## 🏗️ 项目结构

```
transformer-assume/
├── 📁 GPT_LM/                      # GPT风格语言模型实现
│   ├── transformer.py             # 核心Transformer模型
│   ├── moe_transformer.py         # MoE变体实现
│   ├── transformer_block.py       # Transformer块
│   ├── moe_transformer_block.py   # MoE Transformer块
│   ├── attention.py               # 注意力机制
│   ├── train.py                   # 训练脚本
│   ├── inference.py               # 推理脚本
│   ├── tokenizer.py               # 分词器
│   ├── dataset_process.py         # 数据处理
│   └── ... (其他组件)
├── 📁 ed-transformer_py/           # 编码器-解码器Transformer
│   ├── transformer.py             # 完整Transformer模型
│   ├── encoder.py                 # 编码器
│   ├── decoder.py                 # 解码器
│   ├── multihead_attn.py          # 多头注意力
│   ├── dataset.py                 # 数据处理
│   ├── train.py                   # 训练脚本
│   └── evaluation.py              # 评估脚本
├── 📁 notebooks/                   # Jupyter学习笔记
├── 📁 checkpoints/                 # 模型检查点
└── 📄 README.md                   # 项目说明
```

## 🔧 核心实现

### 1. 标准 Transformer (ed-transformer_py/)

基于经典论文 "Attention Is All You Need" 的完整实现：

- **编码器-解码器架构**：适用于序列到序列任务
- **多头注意力机制**：并行学习不同表示子空间
- **位置编码**：使用正弦/余弦函数编码位置信息
- **残差连接 & 层归一化**：稳定深度网络训练

### 2. GPT风格语言模型 (GPT_LM/)

自回归语言模型实现：

- **仅解码器架构**：适用于文本生成任务
- **KV Cache优化**：加速推理过程
- **RoPE位置编码**：相对位置编码方案
- **RMSNorm归一化**：替代传统LayerNorm

### 3. MoE变体 (GPT_LM/moe_*.py)

Mixture of Experts 稀疏激活模型：

- **Top-K路由机制**：动态选择专家网络
- **负载均衡损失**：防止专家使用不均
- **混合架构**：灵活配置MoE层分布
- **辅助损失聚合**：优化训练稳定性

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 2.0
numpy
matplotlib (可选)
jupyter (可选)
```

### 基础使用示例

#### 1. 训练标准GPT模型

```bash
cd GPT_LM
python train.py \
    --train_data_path data/train.bin \
    --valid_data_path data/val.bin \
    --d_model 512 \
    --n_head 8 \
    --n_layer 6 \
    --batch_size 8
```

#### 2. 使用MoE模型

```python
from moe_transformer import MoETransformerLM
import torch

# 创建MoE模型
model = MoETransformerLM(
    d_model=512,
    n_head=8,
    vocab_size=30000,
    max_seq_len=1024,
    n_layer=12,
    n_experts=8,      # 每层8个专家
    top_k=2,          # 激活2个专家
)

# 前向传播
tokens = torch.randint(0, 30000, (4, 128))
logits = model(tokens)
```

#### 3. 机器翻译训练

```bash
cd ed-transformer_py
python train.py
```

## 📊 技术特点

### 模型架构

| 组件 | 实现特点 | 学习价值 |
|------|----------|----------|
| **注意力机制** | 多头自注意力、交叉注意力 | 理解注意力核心原理 |
| **位置编码** | 绝对位置编码、RoPE相对编码 | 掌握位置信息处理 |
| **归一化** | LayerNorm、RMSNorm | 理解归一化技术演进 |
| **激活函数** | SwiGLU、SiLU | 学习现代激活函数 |
| **优化器** | 自定义AdamW实现 | 深入理解优化算法 |

### 实验功能

- **消融实验**：可配置不同组件进行对比
- **梯度裁剪**：防止梯度爆炸
- **学习率调度**：预热+余弦退火
- **混合精度训练**：节省显存提升速度
- **检查点管理**：支持训练恢复

## 🎓 学习路径

### 初学者路径

1. **理论基础**：阅读相关论文和技术博客
2. **代码阅读**：从简单组件开始理解
3. **实验调参**：尝试不同超参数组合
4. **功能扩展**：基于现有代码添加新特性

### 建议学习顺序

```
1. 注意力机制 (attention.py)
2. 位置编码 (rope.py, emb.py)
3. 归一化层 (layernorm.py, rmsnorm.py)
4. Transformer块 (transformer_block.py)
5. 完整模型 (transformer.py)
6. MoE扩展 (moe_*.py)
7. 训练流程 (train.py)
```

## ⚠️ 免责声明

1. **教育用途**：本项目仅用于个人学习和教育目的
2. **代码质量**：代码未经生产环境验证，可能存在bug
3. **性能保证**：不保证模型性能和训练效果
4. **安全风险**：使用者需自行承担使用风险
5. **知识产权**：本项目借鉴了开源社区的研究成果

## 🔗 参考资源

### 核心论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) - MoE实现参考
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - 稀疏专家模型

### 学习资源

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)

## 📄 许可证

**本项目仅限学习和研究使用，禁止任何形式的商业应用。**

---

**⚠️ 再次提醒：这是一个学习项目，不具备商业使用价值！**

如果这个项目对你的学习有帮助，欢迎点个⭐支持一下！