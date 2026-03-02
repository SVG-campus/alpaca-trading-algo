# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(norm_x + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_heads
        self.n_embd = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        
        # RoPE: Precompute for full head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def apply_rotary_pos_emb(self, x, sin, cos):
        # x: [B, H, T, 64] -> Split to [32], [32]
        d_2 = x.shape[-1] // 2
        x1, x2 = x[..., :d_2], x[..., d_2:]
        
        # Split sin/cos to match: [B, 1, T, 64] -> [32], [32]
        c1, c2 = cos[..., :d_2], cos[..., d_2:]
        s1, s2 = sin[..., :d_2], sin[..., d_2:]
        
        # Rotation
        return torch.cat([x1 * c1 - x2 * s1, x1 * s1 + x2 * c1], dim=-1)

    def forward(self, x, input_pos=None, kv_cache=None, mask_idx=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # RoPE Logic
        if input_pos is not None:
            # Inference Mode (Static)
            # input_pos is [B, T] or [1]
            t = input_pos.view(-1)
        else:
            # Training Mode (Dynamic)
            t = torch.arange(T, device=x.device)
            
        freqs = torch.outer(t, self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Reshape for broadcasting [1, 1, T, 64] or [T, 64]
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        
        # Handle broadcasting carefully for both training (3D) and inference (4D)
        if cos.dim() == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        
        q = self.apply_rotary_pos_emb(q, sin, cos)
        k = self.apply_rotary_pos_emb(k, sin, cos)
        
        # KV Cache Handling
        if kv_cache is not None:
            # input_pos is required for static cache
            idx = input_pos.view(-1)
            k_cache, v_cache = kv_cache
            
            # Write to cache
            k_cache[:B, :, idx, :] = k
            v_cache[:B, :, idx, :] = v
            
            # Read from cache (Full buffer for attention)
            k, v = k_cache, v_cache

        # Masking
        attn_mask = None
        if mask_idx is not None and input_pos is not None:
             # Static Inference Masking
            valid_mask = mask_idx <= input_pos.view(B, 1, T, 1)
            attn_mask = torch.where(valid_mask, 0.0, float('-inf'))
        elif input_pos is None:
            # Training Causal Mask
            attn_mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)

        # Attention
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False if attn_mask is not None else True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class TitanLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': RMSNorm(cfg.d_model),
                'attn': CausalSelfAttention(cfg),
                'norm2': RMSNorm(cfg.d_model),
                'ffn': SwiGLU(cfg.d_model, int(8/3 * cfg.d_model))
            }) for _ in range(cfg.n_layers)
        ])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight 
        
        self.max_seq_len = getattr(cfg, 'max_seq_len', 512)
        # Register mask buffer for static inference
        self.register_buffer("mask_idx", torch.arange(self.max_seq_len).view(1, 1, 1, -1))

    def forward(self, x, targets=None, input_pos=None, kv_cache=None):
        x = self.tok_emb(x)
        
        for i, layer in enumerate(self.layers):
            normed_x = layer['norm1'](x)
            # Pass cache for specific layer
            layer_cache = kv_cache[i] if kv_cache is not None else None
            
            attn_out = layer['attn'](normed_x, input_pos, layer_cache, self.mask_idx)
            x = x + attn_out
            x = x + layer['ffn'](layer['norm2'](x))
            
        x = self.norm_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
