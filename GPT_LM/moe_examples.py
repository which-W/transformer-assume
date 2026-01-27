"""
MoE Transformer 使用示例
演示如何使用MoE模型进行训练和推理
"""
import torch
from moe_transformer import MoETransformerLM, HybridMoETransformerLM


def example_basic_moe():
    """基础MoE模型示例"""
    print("=" * 60)
    print("示例1: 基础MoE Transformer模型")
    print("=" * 60)
    
    # 模型配置
    config = {
        "d_model": 512,
        "n_head": 8,
        "vocab_size": 10000,
        "max_seq_len": 1024,
        "d_ff": 2048,
        "theta": 10000.0,
        "n_layer": 12,
        "n_experts": 8,
        "top_k": 2,
        "use_moe_aux_loss": True,
        "moe_aux_loss_weight": 0.01,
    }
    
    # 创建模型
    model = MoETransformerLM(**config)
    
    # 打印模型信息
    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)
    
    print(f"\n模型配置:")
    print(f"  - 层数: {config['n_layer']}")
    print(f"  - 专家数/层: {config['n_experts']}")
    print(f"  - Top-K: {config['top_k']}")
    print(f"  - 模型维度: {config['d_model']}")
    print(f"  - FFN维度: {config['d_ff']}")
    print(f"\n参数统计:")
    print(f"  - 总参数: {total_params:,}")
    print(f"  - 非embedding参数: {non_emb_params:,}")
    
    # 前向传播示例
    batch_size = 4
    seq_len = 128
    token_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    print(f"\n前向传播:")
    print(f"  - 输入shape: {token_ids.shape}")
    
    model.eval()
    with torch.no_grad():
        logits = model(token_ids)
        aux_loss = model.get_aux_loss()
    
    print(f"  - 输出shape: {logits.shape}")
    print(f"  - 辅助损失: {aux_loss.item():.6f}")
    
    return model


def example_hybrid_moe():
    """混合MoE模型示例"""
    print("\n" + "=" * 60)
    print("示例2: 混合MoE Transformer模型")
    print("=" * 60)
    
    config = {
        "d_model": 512,
        "n_head": 8,
        "vocab_size": 10000,
        "max_seq_len": 1024,
        "d_ff": 2048,
        "theta": 10000.0,
        "n_layer": 12,
        "moe_layer_indices": [2, 5, 8, 11],  # 第2,5,8,11层使用MoE
        "n_experts": 8,
        "top_k": 2,
        "use_moe_aux_loss": True,
        "moe_aux_loss_weight": 0.01,
    }
    
    model = HybridMoETransformerLM(**config)
    
    print(f"\n混合架构配置:")
    model.print_architecture()
    
    # 前向传播
    batch_size = 4
    seq_len = 128
    token_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        logits = model(token_ids)
        aux_loss = model.get_aux_loss()
    
    print(f"\n推理结果:")
    print(f"  - 输出shape: {logits.shape}")
    print(f"  - 辅助损失: {aux_loss.item():.6f}")
    
    return model


def example_training_loop():
    """训练循环示例"""
    print("\n" + "=" * 60)
    print("示例3: MoE模型训练循环")
    print("=" * 60)
    
    # 小模型用于演示
    config = {
        "d_model": 256,
        "n_head": 4,
        "vocab_size": 5000,
        "max_seq_len": 512,
        "d_ff": 1024,
        "theta": 10000.0,
        "n_layer": 6,
        "n_experts": 4,
        "top_k": 2,
        "use_moe_aux_loss": True,
        "moe_aux_loss_weight": 0.01,
    }
    
    model = MoETransformerLM(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("\n训练配置:")
    print(f"  - 优化器: AdamW")
    print(f"  - 学习率: 1e-4")
    print(f"  - MoE辅助损失权重: {config['moe_aux_loss_weight']}")
    
    # 模拟训练数据
    batch_size = 2
    seq_len = 64
    
    print(f"\n开始训练演示 (3步)...")
    
    model.train()
    for step in range(3):
        # 生成随机数据
        token_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
        targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
        
        # 前向传播
        logits = model(token_ids)
        
        # 计算语言模型损失
        lm_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config['vocab_size']),
            targets.reshape(-1)
        )
        
        # 获取MoE辅助损失
        aux_loss = model.get_aux_loss()
        
        # 总损失 = 语言模型损失 + MoE辅助损失
        total_loss = lm_loss + aux_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"  Step {step + 1}:")
        print(f"    - LM Loss: {lm_loss.item():.4f}")
        print(f"    - Aux Loss: {aux_loss.item():.6f}")
        print(f"    - Total Loss: {total_loss.item():.4f}")
    
    print("\n训练完成!")
    return model


def example_inference():
    """推理示例"""
    print("\n" + "=" * 60)
    print("示例4: MoE模型推理")
    print("=" * 60)
    
    config = {
        "d_model": 256,
        "n_head": 4,
        "vocab_size": 5000,
        "max_seq_len": 512,
        "d_ff": 1024,
        "theta": 10000.0,
        "n_layer": 6,
        "n_experts": 4,
        "top_k": 2,
    }
    
    model = MoETransformerLM(**config)
    model.eval()
    
    print("\n推理模式特点:")
    print("  - model.eval() 关闭dropout等训练特性")
    print("  - 辅助损失不会被计算(仅训练时需要)")
    print("  - 可以使用torch.no_grad()加速")
    
    # 推理
    batch_size = 1
    seq_len = 32
    token_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(token_ids)
        # 获取预测的token
        predicted_tokens = logits.argmax(dim=-1)
    
    print(f"\n推理结果:")
    print(f"  - 输入shape: {token_ids.shape}")
    print(f"  - 输出logits shape: {logits.shape}")
    print(f"  - 预测tokens shape: {predicted_tokens.shape}")
    
    return model


def compare_moe_vs_standard():
    """对比MoE和标准模型"""
    print("\n" + "=" * 60)
    print("示例5: MoE vs 标准Transformer 对比")
    print("=" * 60)
    
    base_config = {
        "d_model": 512,
        "n_head": 8,
        "vocab_size": 10000,
        "max_seq_len": 1024,
        "d_ff": 2048,
        "theta": 10000.0,
        "n_layer": 12,
    }
    
    # MoE配置
    moe_config = {
        **base_config,
        "n_experts": 8,
        "top_k": 2,
    }
    
    print("\n配置对比:")
    print(f"  基础配置: {base_config['n_layer']}层, d_model={base_config['d_model']}, d_ff={base_config['d_ff']}")
    print(f"  MoE配置: {moe_config['n_experts']}个专家, Top-{moe_config['top_k']}激活")
    
    # 创建MoE模型
    moe_model = MoETransformerLM(**moe_config)
    
    # 参数统计
    moe_params = moe_model.get_num_params(non_embedding=True)
    
    # 每层标准FFN的参数量
    standard_ffn_params = base_config['d_model'] * base_config['d_ff'] * 3  # SwiGLU有3个线性层
    
    # MoE每层的参数量(近似)
    moe_ffn_params = standard_ffn_params * moe_config['n_experts']
    
    print(f"\n参数量对比(单层FFN):")
    print(f"  - 标准FFN: {standard_ffn_params:,}")
    print(f"  - MoE FFN: {moe_ffn_params:,} ({moe_config['n_experts']}个专家)")
    print(f"  - 参数倍增: {moe_ffn_params / standard_ffn_params:.1f}x")
    
    print(f"\n计算量对比:")
    print(f"  - 标准FFN: 每个token都经过完整FFN")
    print(f"  - MoE: 每个token只激活{moe_config['top_k']}个专家")
    print(f"  - 实际计算: {moe_config['top_k']}/{moe_config['n_experts']} = {moe_config['top_k']/moe_config['n_experts']:.1%}")
    
    print(f"\n优势:")
    print(f"  ✓ MoE用更少的计算获得更大的模型容量")
    print(f"  ✓ 参数多{moe_config['n_experts']}倍,但计算只增加{moe_config['top_k']/moe_config['n_experts']:.0%}")
    print(f"  ✓ 不同专家可以学习不同的专业知识")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MoE Transformer 完整示例")
    print("="*60)
    
    # 运行所有示例
    example_basic_moe()
    example_hybrid_moe()
    example_training_loop()
    example_inference()
    compare_moe_vs_standard()
    
    print("\n" + "="*60)
    print("所有示例运行完成!")
    print("="*60)
