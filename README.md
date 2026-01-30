# Transformer-Assume

一个基于PyTorch的Transformer模型实现，专注于德语到英语的机器翻译任务。这是一个学习项目，采用模块化设计，详细实现了Transformer架构的各个组件。

## 🎯 项目概述

本项目从零开始实现了完整的Transformer模型，包括编码器(Encoder)、解码器(Decoder)、多头注意力机制(Multi-head Attention)、位置编码(Positional Encoding)等核心组件。项目使用Multi30k数据集进行德英翻译任务的训练和评估。

## 📁 项目结构

```
transformer-assume/
├── ed-transformer_py/              # 核心Python模块
│   ├── config.py                   # 配置文件
│   ├── dataset.py                  # 数据集处理和词表构建
│   ├── emb.py                      # 词嵌入和位置编码
│   ├── multihead_attn.py           # 多头注意力机制
│   ├── encoder_block.py            # 编码器块
│   ├── encoder.py                  # 编码器
│   ├── decoder_block.py            # 解码器块
│   ├── decoder.py                  # 解码器
│   ├── transformer.py              # 完整Transformer模型
│   ├── train.py                    # 训练脚本
│   └── evaluation.py               # 评估脚本
├── ed_transformer_notebook/        # Jupyter笔记本
│   ├── embeding.ipynb              # 词嵌入实验
│   ├── multihead_atten.ipynb       # 注意力机制实验
│   ├── Encoder.ipynb               # 编码器实验
│   ├── Decoder.ipynb               # 解码器实验
│   ├── Decoder_block.ipynb         # 解码器块实验
│   ├── encoder_block.ipynb         # 编码器块实验
│   ├── transformer.ipynb           # 完整模型实验
│   ├── train.ipynb                 # 训练过程
│   ├── evaluation.ipynb            # 评估过程
│   └── test.ipynb                  # 测试脚本
├── multi30k/                       # 数据集目录
│   ├── train.1.de                  # 德语训练数据
│   └── train.1.en                  # 英语训练数据
├── checkpoints/                    # 模型检查点
│   └── model.pth                   # 训练好的模型
├── requirements.txt                # 依赖包列表
├── LICENSE                         # Apache 2.0许可证
└── README.md                       # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.5.1+ (支持CUDA)
- CUDA 12.1 (推荐，用于GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

1. 下载Multi30k数据集
2. 将德语和英语训练文件放置在 `multi30k/` 目录下
3. 确保文件名为 `train.1.de` 和 `train.1.en`

### 训练模型

```bash
cd ed-transformer_py
python train.py
```

### 评估模型

```bash
cd ed-transformer_py
python evaluation.py
```

## 🧩 模块化设计

### 核心组件

1. **配置模块 (`config.py`)**
   - 设备配置 (CPU/GPU)
   - 序列最大长度设置

2. **数据处理 (`dataset.py`)**
   - 使用spaCy进行德英分词
   - 构建词汇表
   - 特殊token处理 (UNK, PAD, BOS, EOS)
   - 数据预处理管道

3. **嵌入层 (`emb.py`)**
   - 词嵌入实现
   - 位置编码
   - Dropout正则化

4. **注意力机制 (`multihead_attn.py`)**
   - 缩放点积注意力
   - 多头注意力实现
   - 注意力权重可视化

5. **编码器组件**
   - `encoder_block.py`: 编码器块 (自注意力 + 前馈网络)
   - `encoder.py`: 完整编码器 (多层编码器块堆叠)

6. **解码器组件**
   - `decoder_block.py`: 解码器块 (掩码自注意力 + 交叉注意力 + 前馈网络)
   - `decoder.py`: 完整解码器 (多层解码器块堆叠)

7. **完整模型 (`transformer.py`)**
   - 编码器-解码器架构
   - 端到端训练支持

8. **训练和评估**
   - `train.py`: 模型训练脚本
   - `evaluation.py`: 模型评估脚本

### 设计特点

- **模块化**: 每个组件独立实现，便于理解和修改
- **可配置**: 超参数集中管理，易于实验
- **可扩展**: 支持不同规模的模型配置
- **教学友好**: 详细的中文注释和Jupyter实验

## 📊 模型配置

默认模型配置：
- 嵌入维度: 512
- 注意力头数: 8
- 编码器层数: 6
- 解码器层数: 6
- 前馈网络维度: 2048
- Dropout率: 0.1
- 最大序列长度: 5000

## 🎓 学习目标

通过本项目，您可以深入理解：

1. **Transformer架构原理**
   - 自注意力机制
   - 位置编码
   - 编码器-解码器结构

2. **机器翻译任务**
   - 序列到序列建模
   - 德英语言对处理
   - BLEU评估指标

3. **深度学习实践**
   - PyTorch模型开发
   - 批量训练技术
   - 模型保存与加载

4. **代码组织**
   - 模块化编程
   - 配置管理
   - 实验记录

## 🛠️ 技术栈

- **深度学习框架**: PyTorch 2.5.1+
- **自然语言处理**: spaCy
- **数据处理**: NumPy, torch.utils.data
- **实验记录**: Jupyter Notebook
- **版本控制**: Git
- **依赖管理**: pip

## 📈 实验记录

项目包含完整的Jupyter实验记录，涵盖：
- 词嵌入和位置编码实验
- 多头注意力机制可视化
- 编码器和解码器组件测试
- 完整模型训练过程
- 性能评估和结果分析

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用Apache License 2.0许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢Multi30k数据集提供者
- 感谢PyTorch团队提供的优秀框架
- 感谢spaCy团队提供的NLP工具

---

**注意**: 这是一个学习项目，主要用于教学和研究目的。如需在生产环境中使用，建议进行进一步的优化和测试。