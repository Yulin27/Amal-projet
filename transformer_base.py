import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScaledDotProduct(torch.nn.Module):

    def __init__(self, dropout=0.0):
        super(ScaledDotProduct, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, torch.Tensor]:
        
        d_k = query.size(-1)  # 获取query的最后一个维度的大小，用于缩放

        # 计算query和key的点积，然后除以缩放因子 sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 如果有mask，将mask应用到scores上
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        # 应用softmax得到注意力权重，然后应用dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 用注意力权重乘以value
        output = torch.matmul(attn, value)

        return output, attn
    

class MultiHeadAttention(nn.Module):
    
        def __init__(self, d_model, num_heads, dropout=0.0):
            super(MultiHeadAttention, self).__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.dropout = nn.Dropout(dropout)
    
            # 确保d_model可以被num_heads整除
            assert d_model % num_heads == 0
            # 得到head的维度
            self.d_k = d_model // num_heads
    
            # 初始化四个线性层
            self.q_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)
            self.k_linear = nn.Linear(d_model, d_model)
            self.out = nn.Linear(d_model, d_model)
    
            self.attn = ScaledDotProduct(dropout=dropout)
    
        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            
            # 获取batch_size
            batch_size = q.size(0)
    
            # 通过线性层进行映射
            k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k)
            q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k)
            v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k)
    
            # 转置以获得batch_size和num_heads
            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
    
            # 计算注意力
            scores, attn = self.attn(q, k, v, attn_mask=attn_mask)
    
            # 将多头维度连接起来
            concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
            # 通过最后一个线性层
            output = self.out(concat)
    
            return output, attn
        
class encoder_layer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super(encoder_layer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 注意力层
        attn_output, attn = self.attn(x, x, x, attn_mask=attn_mask)
        attn_output = self.dropout(attn_output)
        # 残差连接和归一化
        out1 = self.norm1(x + attn_output)
        # 前馈神经网络
        ff_output = self.ff(out1)
        ff_output = self.dropout(ff_output)
        # 残差连接和归一化
        out2 = self.norm2(out1 + ff_output)
        return out2
    
class ClassificationTransformer(nn.Module):
    def __init__(self,d_input, d_output, d_model, nhead, dropout=0.0, num_layers=1):
        super(ClassificationTransformer, self).__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.d_model = d_model
        self.num_heads = nhead
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.input = nn.Linear(d_input, d_model)
        self.encoder = nn.ModuleList([encoder_layer(d_model, nhead, dropout=dropout) for _ in range(num_layers)])
        self.out = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 通过线性层
        x = self.input(x)
        # 通过多个encoder layer
        for layer in self.encoder:
            x = layer(x, attn_mask=attn_mask)
        # 通过最后一个线性层
        x = self.out(x[:, 0, :])
        return x

