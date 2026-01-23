# Transformer-Assume

简易复现Transformer架构的小白版本，用于学习和理解Transformer模型的基本实现。

## 项目描述

这个项目实现了Transformer模型的核心组件，用于德语到英语的机器翻译任务。通过逐步实现嵌入层、多头注意力机制、编码器、解码器等模块，帮助学习者理解Transformer的工作原理。

## 依赖项

- PyTorch (版本差异说明: .py文件使用PyTorch 2.0以下版本，.ipynb文件使用PyTorch 2.8左右版本)
- NumPy
- Matplotlib (用于可视化)
- Jupyter Notebook (用于运行.ipynb文件)

## 项目结构

```
transformer-assume/
├── py_files/                 # Python 脚本文件夹
│   ├── config.py             # 配置文件，包含序列最大长度和设备设置
│   ├── dataset.py            # 数据集处理和预处理
│   ├── train.py              # 训练脚本，用于训练德语到英语翻译模型
│   ├── evaluation.py         # 评估脚本，用于测试模型性能
│   ├── transformer.py        # 主Transformer模型类
│   ├── encoder.py            # 编码器实现
│   ├── decoder.py            # 解码器实现
│   ├── encoder_block.py      # 编码器块
│   ├── decoder_block.py      # 解码器块
│   ├── multihead_attn.py     # 多头注意力机制
│   ├── emb.py                # 嵌入层实现
│   ├── emb_2.py              # 嵌入层的另一种实现
│   └── Linner.py             # 线性层相关代码
├── notebook_files/           # Jupyter notebooks文件夹
│   ├── embeding.ipynb        # 嵌入层演示
│   ├── multihead_atten.ipynb # 多头注意力机制演示
│   ├── Encoder.ipynb         # 编码器演示
│   ├── Decoder.ipynb         # 解码器演示
│   ├── transformer.ipynb     # 完整Transformer模型演示
│   ├── train.ipynb           # 训练过程演示
│   ├── evaluation.ipynb      # 评估过程演示
│   └── ...                   # 其他notebook文件
├── checkpoints/              # 模型检查点
│   └── model.pth            # 保存的模型权重
├── multi30k/                 # 数据集文件夹 (Multi30k数据集)
│   ├── train.1.de           # 德语训练数据
│   └── train.1.en           # 英语训练数据
├── __pycache__/             # Python缓存文件
├── .ipynb_checkpoints/      # Jupyter notebook检查点
├── .gitignore               # Git忽略文件
├── LICENSE                  # 许可证
└── README.md                # 项目说明文档
```

## 主要文件说明

### 核心模型文件
- **py_files/transformer.py**: 主Transformer模型，包含编码器和解码器
- **py_files/encoder.py**: Transformer编码器实现
- **py_files/decoder.py**: Transformer解码器实现
- **py_files/multihead_attn.py**: 多头注意力机制的实现
- **py_files/emb.py**: 词嵌入和位置编码的实现

### 数据处理
- **py_files/dataset.py**: 数据集加载、词汇表构建、分词预处理
- **multi30k/**: Multi30k数据集，用于德语-英语翻译任务

### 训练和评估
- **py_files/train.py**: 模型训练脚本，使用SGD优化器训练翻译模型
- **py_files/evaluation.py**: 模型评估脚本

### 配置
- **py_files/config.py**: 项目配置，包括最大序列长度和计算设备设置

### Jupyter Notebooks
项目包含多个Jupyter notebooks，用于逐步演示各个组件：
- **notebook_files/embeding.ipynb**: 嵌入层演示
- **notebook_files/multihead_atten.ipynb**: 多头注意力机制演示
- **notebook_files/Encoder.ipynb**: 编码器演示
- **notebook_files/Decoder.ipynb**: 解码器演示
- **notebook_files/transformer.ipynb**: 完整Transformer模型演示
- **notebook_files/train.ipynb**: 训练过程演示
- **notebook_files/evaluation.ipynb**: 评估过程演示

## 如何运行

### 环境准备
1. 安装PyTorch (根据您的环境选择版本)
   ```bash
   pip install torch torchvision torchaudio
   ```

2. 安装其他依赖
   ```bash
   pip install numpy matplotlib jupyter
   ```

### 训练模型
运行训练脚本：
```bash
python py_files/train.py
```

### 测试模型
运行评估脚本：
```bash
python py_files/evaluation.py
```

### Jupyter Notebook演示
启动Jupyter并打开相应的notebook文件：
```bash
jupyter notebook
```

## 模型参数

在`train.py`中定义的模型参数：
- 嵌入维度: 512
- Q/K维度: 64
- V维度: 64
- 前馈网络维度: 2048
- 注意力头数: 8
- 编码器/解码器层数: 6
- Dropout: 0.1
- 最大序列长度: 5000

## 数据集

使用Multi30k数据集进行训练，该数据集包含德语-英语平行语料，用于机器翻译任务。

## 许可证

请查看LICENSE文件了解项目许可证信息。

## 贡献

欢迎提交Issue和Pull Request来改进这个学习项目。

## 注意事项

- 这个实现是学习目的的简化版本，可能与原始Transformer论文有一些差异
- 建议先通过Jupyter notebooks了解各个组件，然后运行训练脚本
- 训练过程可能需要较长时间，取决于硬件配置