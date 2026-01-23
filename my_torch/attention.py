import torch 
import math 
from softmax import StableSoftmax
from torch import nn
from einops import rearrange
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
    probs = StableSoftmax(scores, dim=-1)
    
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
        self_d_k = d_model // n_head
        self.device = device
        #Q,K,V投影层，投影到各自的维度
        factory_par = {"device":device , "dtype":dtype}
        self.q_pro = nn.Linear(d_model , d_model ,**factory_par)
        self.k_pro = nn.Linear(d_model , d_model ,**factory_par)
        self.v_pro = nn.Linear(d_model , d_model ,**factory_par)
        
        self.output_pro = nn.Linear(d_model , d_model ,**factory_par)
        
        if theta is not None and max_seq_size is not None:
            self.rope = RotaryPositionalEmbedding(theta,self_d_k,max_seq_size,device=device)
        else:
            self.rope = None
        
    def forward(self,x:torch.tensor, token_position:torch.tensor = None):
        b,s,d = x.shape
        #将映射拆分为多头
        q = rearrange(self.q_pro(x),'... s (h d) -> ... h s d')
        k = rearrange(self.k_pro(x),'... s (h d) -> ... h s d')
        v = rearrange(self.v_pro(x),'... s (h d) -> ... h s d')

        #应用RoPE 旋转位置编码
        
        
        
        #生成因果掩码，保证query只能看见当前以及之前的key
        mask = torch.tril(torch.ones(s,s,device = self.device,dtype = torch.long))
        
        #核心注意力计算（SDPA）(bath_size,heads,seq,d_k)
        attn_out = scaled_dot_product_attention(q,k,v,mask=mask)
        
        #合并多头
        attn_out = rearrange(attn_out,'... h s d -> ... s (h d)')

        return self.output_pro(attn_out)