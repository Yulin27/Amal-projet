"""Wrapper around nn.MultiheadAttention to adhere to SequenceModule interface."""

import torch
import torch.nn.functional as F
from torch import nn
import hydra
from models.sequence.base import SequenceModule, TransposedModule
import src.models.nn.utils as U
from einops import rearrange
import math
@TransposedModule
class MultiheadAttention(SequenceModule):
    """Simple wrapper for MultiheadAttention."""
    def __init__(self, d_model, n_heads, *args, causal=True, use_grad_cp=False, chunk_size=0,pe_heads=False , ablate_data_dep=False, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size
        self.ablate_data_dep = ablate_data_dep
        self.pe_d_model = d_model // 4 if not ablate_data_dep else d_model
        self.use_grad_cp = use_grad_cp
        self.d_model = d_model
        self.d_output = d_model 
        self.pe_heads = pe_heads
        from src.models.baselines.transformer import MultiheadAttention
        #self.mha = nn.MultiheadAttention(d_model, n_heads, *args, batch_first=True, **kwargs)
        self.mha = MultiheadAttention(d_model, n_heads, causal=causal, *args, **kwargs)  
        if self.pe_heads:  
            self.gate= False
            import copy
            kwargs2 = copy.deepcopy(kwargs)
            kwargs2["vdim"] = self.d_model
            self.positional_mha = MultiheadAttention(self.d_model, n_heads, causal=causal,q_in_dim=self.pe_d_model, *args, **kwargs2) 
            if self.gate:
                self.alpha = torch.nn.Parameter(-torch.log(torch.rand(self.pe_d_model)))   
            else:
                self.mixing_linear = torch.nn.Linear(d_model+self.d_model,d_model)
        self.causal = causal
        print("Create MultiheadAttention layer with causal=",causal,"| args:", args)
    
    def checkpoint_mha(self, q, k, v, attn_mask, key_padding_mask):
        return self.mha(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)

    
    def forward(self, src, attn_mask=None, key_padding_mask=None, state=None, **kwargs):
        if self.chunk_size >0 :
            seq_len = src.size(-2)
            pad_amount = (self.chunk_size - (seq_len % self.chunk_size)) % self.chunk_size  
            padded_seq_len = seq_len + pad_amount  
            if pad_amount > 0: 
                padding = src.new_zeros((src.size(0), pad_amount, src.size(-1)))  
                src = torch.cat((src, padding), dim=1)  
            
        if self.causal and attn_mask is None:
            attn_mask = torch.triu(torch.ones(src.size(-2), src.size(-2),
                                              dtype=torch.bool, device=src.device),
                                       diagonal=1)
            if self.chunk_size >0:
                attn_mask = attn_mask[:self.chunk_size,:self.chunk_size]

        # attn_mask, key_padding_mask = state
        # Note that this returns None for the second argument
        src = src.permute(1,0,2)
        if self.ablate_data_dep:
            seq_len = src.size(0)
            pe = self.generate_positional_encoding(seq_len, self.pe_d_model, src.device)
            pe = pe.unsqueeze(1)
            y, _ = self.mha(pe, pe, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False,pe_based=True, chunk_size=self.chunk_size)
        elif not self.use_grad_cp:
            y, _ = self.mha(src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, chunk_size=self.chunk_size)
            if self.chunk_size >0 and pad_amount > 0: 
                y = y[:-pad_amount, :, :]  
        else:
            y, _ = torch.utils.checkpoint.checkpoint(
            self.checkpoint_mha,
            src, src, src, attn_mask, key_padding_mask
        )
        if self.pe_heads:
            seq_len = src.size(0)
            pe = self.generate_positional_encoding(seq_len, self.pe_d_model, src.device)
            pe = pe.unsqueeze(1)
            y_pos, _ = self.positional_mha(pe, pe, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False,pe_based=True)
            if self.gate:
                y = self.alpha * y_pos + (1-self.alpha)*y
            else:
                y = torch.cat([y,y_pos],dim=-1)
                y = self.mixing_linear(y)

        return y.permute(1,0,2), None
    
    @staticmethod
    def generate_positional_encoding(seq_len, d_model,device):
        # Assuming positional encoding is sine-cosine based
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.to(device)  # shape: (seq_len, d_model)
    
    def step(self, x, state):
        # TODO proper cached inference
        # x: (B, D)
        y, z = self.mha(src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, **kwargs)


class VitAttention(SequenceModule):
    """Copied from implementation for ViT: only used for ViT model.

    This attention class makes several simplifying assumptions (commonly satisfied in vision
       applications):
    1. q = k = v
    2. No masks: no attention mask, no key padding mask
    3. Embed dimension = Input dimension, i.e. projection matrices are square.

    Arguments:
    - packed_linear: whether to pack all 3 q_proj, k_proj, v_proj into 2 matrix.
        This option is to be compatible with T2T-ViT pretrained weights,
        where there's only one projection weight matrix.
    """

    @property
    def d_output(self):
        return self.dim

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        # proj_drop=0.,
        packed_linear=True,
        linear_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        if linear_cfg is not None:
            packed_linear = False
        self.packed_linear = packed_linear
        if packed_linear:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            if linear_cfg is None:
                linear_cfg = {'_target_': 'torch.nn.Linear'}
            self.q_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)
            self.k_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)
            self.v_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        # Removing this dropout because we do this in SequenceResidualBlock
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, state=None):
        B, N, C = x.shape
        if self.packed_linear:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            q, k, v = [rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads) for x in (q, k, v)]

        # attn = (q @ k.transpose(-2, -1) * self.scale)
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = q.size()
        _, _, k_seq_len, _ = k.size()
        q = rearrange(q, 'b h t d -> (b h) t d')
        k = rearrange(k, 'b h s d -> (b h) d s')
        # Preallocate attn_weights for `baddbmm`
        attn = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=q.dtype, device=q.device)
        attn = rearrange(torch.baddbmm(attn, q, k, beta=0, alpha=self.scale),
                         '(b h) t s -> b h t s', h = self.num_heads)

        attn = F.softmax(attn, dim=-1, dtype=v.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x, None
