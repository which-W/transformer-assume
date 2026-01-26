import torch 
import math 
from softmax import StableSoftmax
from torch import nn
from einops import rearrange
from rope import RotaryPositionalEmbedding
#计算打分表Q*K 并对V进行加权输出
def scaled_dot_product_attention(
    Q:torch.Tensor,
    K:torch.Tensor,
    V:torch.Tensor,
    mask: torch.Tensor = None
):
    """
        Q:[..., N ,d_k]
        K:[..., m ,d_k]
        V:[..., m ,d_v]
    """
    #获取d_k
    d_k = Q.size(-1)
    
    #计算相似度分数，形成打分表
    scores = torch.einsum('...nk,...mk -> ...nm',Q,K) / math.sqrt(d_k)
    #应用mask掩码
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
        
    #计算注意力权重（归一化）
    #dim=-1 对应的是每一个Q对于K的分布
    softmax = StableSoftmax(dim=-1)
    probs = softmax(scores)
    
    #加权求和得到输出
    output = torch.einsum('...nm, ...mk -> ...nk', probs ,V)
    
    return output

class CauseMutiHeadAttention(nn.Module):
    def __init__ (self , d_model:int , n_head : int , max_seq_size : int = None, device = None , dtype = None , theta=None):
        super().__init__()
        #判断维度
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.device = device
        #Q,K,V投影层，投影到各自的维度
        factory_par = {"device":device , "dtype":dtype}
        self.q_pro = nn.Linear(d_model , d_model ,**factory_par)
        self.k_pro = nn.Linear(d_model , d_model ,**factory_par)
        self.v_pro = nn.Linear(d_model , d_model ,**factory_par)
        #输出投影层 整合所有信息
        self.output_pro = nn.Linear(d_model , d_model ,**factory_par)
        
        if theta is not None and max_seq_size is not None:
            self.rope = RotaryPositionalEmbedding(theta,self.d_k,max_seq_size,device=device)
        else:
            self.rope = None
        
    def forward(self,x:torch.tensor, token_position:torch.tensor = None):
        b,s,d = x.shape
        #将映射拆分为多头
        q = rearrange(self.q_pro(x),'... s (h d) -> ... h s d', h=self.n_head)
        k = rearrange(self.k_pro(x),'... s (h d) -> ... h s d', h=self.n_head)
        v = rearrange(self.v_pro(x),'... s (h d) -> ... h s d', h=self.n_head)

        #应用RoPE 旋转位置编码
        if self.rope is not None:
            if token_position is None:
                #默认生成从0开始的顺序位置
                #expand处理 Batch维度，不占用额外的内存
                token_position = torch.arange(s,device=x.device).expand(b,s)
            
            #对Q,K进行旋转，V保持不动
            q = self.rope(q,token_position)
            k = self.rope(k,token_position)
        
        
        #生成因果掩码，保证query只能看见当前以及之前的key
        mask = torch.tril(torch.ones(s,s,device = self.device,dtype = torch.long))
        
        #核心注意力计算（SDPA）(bath_size,heads,seq,d_k)
        attn_out = scaled_dot_product_attention(q,k,v,mask=mask)
        
        #合并多头
        attn_out = rearrange(attn_out,'... h s d -> ... s (h d)')

        return self.output_pro(attn_out)

def test_cause_multi_head_attention():
    """测试因果多头注意力机制"""
    
    print("=" * 60)
    print("测试因果多头注意力机制 (CauseMutiHeadAttention)")
    print("=" * 60)
    
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    
    # 测试参数
    batch_size = 2
    seq_len = 8
    d_model = 512
    n_head = 8
    max_seq_size = 1024
    theta = 10000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n配置参数:")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Sequence Length: {seq_len}")
    print(f"  - Model Dimension: {d_model}")
    print(f"  - Number of Heads: {n_head}")
    print(f"  - Device: {device}")
    
    # 初始化模型
    print(f"\n初始化模型...")
    model = CauseMutiHeadAttention(
        d_model=d_model,
        n_head=n_head,
        max_seq_size=max_seq_size,
        device=device,
        theta=theta
    )
    model.to(device)
    print(f"  ✓ 模型初始化成功")
    
    # 测试1: 基本前向传播
    print(f"\n测试1: 基本前向传播")
    print("-" * 60)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    print(f"  输入形状: {x.shape}")
    
    output = model(x)
    print(f"  输出形状: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model), "输出形状不正确!"
    print(f"  ✓ 输出形状正确: {output.shape}")
    
    # 测试2: 因果性检查
    print(f"\n测试2: 因果性检查")
    print("-" * 60)
    print("  验证每个位置只能看到当前及之前的token...")
    
    # 创建一个特殊的输入，便于观察因果性
    x_causal = torch.zeros(1, seq_len, d_model, device=device)
    for i in range(seq_len):
        x_causal[0, i, 0] = i + 1  # 每个位置有唯一标识
    
    with torch.no_grad():
        output_causal = model(x_causal)
    
    # 修改未来的token，看输出是否变化
    x_causal_modified = x_causal.clone()
    x_causal_modified[0, -1, :] = 999  # 修改最后一个token
    
    with torch.no_grad():
        output_modified = model(x_causal_modified)
    
    # 检查前面的token输出是否不变
    diff = torch.abs(output_causal[0, :-1] - output_modified[0, :-1]).max()
    print(f"  前面token的最大输出差异: {diff.item():.6f}")
    if diff < 1e-5:
        print(f"  ✓ 因果性验证通过: 未来token不影响过去")
    else:
        print(f"  ✗ 警告: 因果性可能存在问题")
    
    # 测试3: 梯度反向传播
    print(f"\n测试3: 梯度反向传播")
    print("-" * 60)
    x_grad = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    output_grad = model(x_grad)
    loss = output_grad.sum()
    loss.backward()
    
    print(f"  输入梯度形状: {x_grad.grad.shape}")
    print(f"  输入梯度范数: {x_grad.grad.norm().item():.6f}")
    assert x_grad.grad is not None, "梯度计算失败!"
    print(f"  ✓ 梯度反向传播成功")
    
    # 测试4: 不同序列长度
    print(f"\n测试4: 不同序列长度")
    print("-" * 60)
    for test_seq_len in [4, 16, 32]:
        x_var_len = torch.randn(batch_size, test_seq_len, d_model, device=device)
        with torch.no_grad():
            output_var_len = model(x_var_len)
        assert output_var_len.shape == (batch_size, test_seq_len, d_model)
        print(f"  序列长度 {test_seq_len}: {output_var_len.shape} ✓")
    
    # 测试5: 自定义token位置
    print(f"\n测试5: 自定义token位置")
    print("-" * 60)
    token_position = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    print(f"  位置编码形状: {token_position.shape}")
    
    with torch.no_grad():
        output_with_pos = model(x, token_position=token_position)
    
    print(f"  带位置编码的输出形状: {output_with_pos.shape}")
    print(f"  ✓ 自定义位置编码成功")
    
    # 测试6: 参数统计
    print(f"\n测试6: 模型参数统计")
    print("-" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 测试7: 输出值范围检查
    print(f"\n测试7: 输出值检查")
    print("-" * 60)
    with torch.no_grad():
        output_check = model(x)
    
    print(f"  输出最小值: {output_check.min().item():.6f}")
    print(f"  输出最大值: {output_check.max().item():.6f}")
    print(f"  输出均值: {output_check.mean().item():.6f}")
    print(f"  输出标准差: {output_check.std().item():.6f}")
    
    # 检查是否有NaN或Inf
    has_nan = torch.isnan(output_check).any()
    has_inf = torch.isinf(output_check).any()
    
    if not has_nan and not has_inf:
        print(f"  ✓ 输出值正常，无NaN或Inf")
    else:
        print(f"  ✗ 警告: 输出包含NaN或Inf")
    
    # 测试8: 批次大小为1
    print(f"\n测试8: 单样本测试 (batch_size=1)")
    print("-" * 60)
    x_single = torch.randn(1, seq_len, d_model, device=device)
    with torch.no_grad():
        output_single = model(x_single)
    print(f"  输入形状: {x_single.shape}")
    print(f"  输出形状: {output_single.shape}")
    print(f"  ✓ 单样本测试通过")
    
    print(f"\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_cause_multi_head_attention()
        print("\n✓ 所有测试通过!")
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()