#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectralDINO: Dual Mixture-of-Subspaces (Dual-MoS) LoRA for Hyperspectral Image Classification (HyRank)
Architecture:
- Vision Transformer backbone (ViT-Small configuration: 12 blocks, 384 embed_dim, 6 heads)
- SpectralTokenizer: 1x1 conv spectral projection (bands * 256 params) + 16 spectral-spatial tokens
- Dual-MoS LoRA: Multi-expert low-rank adaptation (rank=4) across spectral and spatial subspaces
- DINO Self-Distillation projection head + feature extractor for OSR / downstream classification
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Dual Mixture-of-Subspaces (Dual-MoS) LoRA Linear Layer.
    Implements multi-expert subspace low-rank decomposition:
      W_adapted(x) = W_base(x) + (alpha / r) * sum_i g_i(x) * (B_i @ A_i @ x)
    where g(x) is a learned softmax gating router over dual subspace experts:
      Expert 1: Spectral-domain adaptation
      Expert 2: Spatial-contextual adaptation
    """
    def __init__(self, in_features, out_features, rank=4, num_experts=2, alpha=8.0, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_experts = num_experts
        self.scaling = alpha / rank
        
        self.linear = nn.Linear(in_features, out_features)
        
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(rank, in_features)) for _ in range(num_experts)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(out_features, rank)) for _ in range(num_experts)
        ])
        
        self.router = nn.Linear(in_features, num_experts)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        for a in self.lora_A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        for b in self.lora_B:
            nn.init.zeros_(b)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        
    def forward(self, x):
        base_out = self.linear(x)
        gate_logits = self.router(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        lora_out = torch.zeros_like(base_out)
        dropped_x = self.dropout(x)
        for i in range(self.num_experts):
            sub_act = F.linear(dropped_x, self.lora_A[i])
            sub_proj = F.linear(sub_act, self.lora_B[i])
            lora_out = lora_out + sub_proj * gate_weights[..., i:i+1]
            
        return base_out + self.scaling * lora_out


class DualMoSAttention(nn.Module):
    """Multi-Head Self-Attention with Dual-MoS LoRA projection layers."""
    def __init__(self, embed_dim=384, num_heads=6, rank=4, num_experts=2, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = LoRALinear(embed_dim, embed_dim, rank=rank, num_experts=num_experts)
        self.k_proj = LoRALinear(embed_dim, embed_dim, rank=rank, num_experts=num_experts)
        self.v_proj = LoRALinear(embed_dim, embed_dim, rank=rank, num_experts=num_experts)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        return out


class MLP(nn.Module):
    """Feedforward network with GELU activation."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm and Dual-MoS Attention."""
    def __init__(self, embed_dim=384, num_heads=6, rank=4, num_experts=2, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = DualMoSAttention(embed_dim, num_heads=num_heads, rank=rank, num_experts=num_experts, dropout=drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpectralTokenizer(nn.Module):
    """
    Spectral-Spatial Tokenizer:
    1. 1x1 Conv to project spectral bands to 256 channels (in_bands * 256 parameters).
    2. Adaptive 4x4 spatial pooling producing 16 spatial tokens.
    3. Linear projection to embed_dim (384).
    4. Learnable CLS token and position embeddings (17 tokens total).
    """
    def __init__(self, in_bands=176, num_tokens=16, embed_dim=384, patch_size=15):
        super().__init__()
        self.in_bands = in_bands
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        
        self.band_proj = nn.Conv2d(in_bands, 256, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.token_proj = nn.Linear(256, embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        feat = self.band_proj(x)
        pooled = self.pool(feat)
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1)
        x = x + self.pos_embed
        return x


class SpectralDINO(nn.Module):
    """
    SpectralDINO: Dual Mixture-of-Subspaces LoRA for Hyperspectral Image Few-Shot Classification.
    """
    def __init__(self, in_bands=176, num_classes=12, embed_dim=384, depth=12, num_heads=6,
                 rank=4, num_experts=2, num_tokens=16, patch_size=15):
        super().__init__()
        self.in_bands = in_bands
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        self.tokenizer = SpectralTokenizer(
            in_bands=in_bands, num_tokens=num_tokens, embed_dim=embed_dim, patch_size=patch_size
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, rank=rank, num_experts=num_experts)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        self.dino_head = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 256),
        )
        
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def extract_features(self, x):
        """Extract L2-normalized 384-dimensional feature representation from CLS token."""
        x = self.tokenizer(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_feat = x[:, 0]
        return F.normalize(cls_feat, p=2, dim=-1)
        
    def forward(self, x, return_features=False):
        x = self.tokenizer(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_feat = x[:, 0]
        
        logits = self.classifier(cls_feat)
        dino_proj = self.dino_head(cls_feat)
        
        if return_features:
            return logits, dino_proj, cls_feat
        return logits
