import torch
from torch import nn
from emb import CustomEmbedding
from rmsnorm import RMSNorm
from transformer_block import TransformerBlock
class TransformerLM(nn.Module):
    def __init__(self,d_model:int,n_head:int,vocab_size:int,
                 max_seq_len:int,d_ff:int,theta:float,n_layer:int,
                 device=None , dtype=None
                 ,#实验参数
                 use_rms_norm:bool = True,
                 norm_model:str = "pre",
                 ffn_type:str = "swiglu", 
                 ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        factroy_pra = {"device":device,"dtype":dtype}
        #初始化embeding层
        self.embedding = CustomEmbedding(vocab_size,d_model,**factroy_pra)
        #堆叠transformer block
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_head=n_head,
                    max_seq_len=max_seq_len,
                    theta=theta,
                    **factroy_pra,
                )
                for _ in range(n_layer)
            ]
        )
        #最终输出层
        #如果全局禁用norm这里的RMSNorm要设为Identity
        if use_rms_norm:
            self.ln_final = RMSNorm(d_model,**factroy_pra)
        else:
            self.ln_final = nn.Identity()
            
        #最后一个Linner用来返回词表大小
        self.ln_output = nn.Linear(d_model,vocab_size,**factroy_pra)
    
    def forward(self,token_ids:torch.Tensor):
        b,s = token_ids.shape()
        #获取Rope的位置向量
        token_position = torch.arange(s,device=self.device,dtype=self.dtype).unsqueeze(0).expand(b,s)
        #embeding
        x = self.embedding(token_ids,device = self.device,dtype=self.dtype)
        #逐层通过block
        for layer in self.layers:
            x = layer(x,token_position,device = self.device,dtype=self.dtype)
        #最终归一化，如果use_rms_norm为false则不会通过这一层
        x = self.ln_final(x)
        #返回投射到词表空间的logits
        return self.ln_output(x)