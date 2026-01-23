import torch 
import math 
from softmax import StableSoftmax

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
    probs = StableSoftmax(scores, dim=-1)
    
    #加权求和得到输出
    output = torch.einsum('...nm, ...mk -> ...nk', probs ,V)
    
    return output