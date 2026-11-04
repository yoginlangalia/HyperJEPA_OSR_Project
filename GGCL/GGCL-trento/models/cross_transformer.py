"""
GGCL Cross Transformer Model
Based on: "Enhancing Hyperspectral Image Classification: Leveraging Unsupervised 
Information With Guided Group Contrastive Learning"
Authors: Ben Li, Leyuan Fang, Ning Chen, Jitong Kang, Jun Yue
IEEE TGRS 2024
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    """Pre-normalization layer"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    """Feed-forward network"""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, dim, heads=8, droupout=0.1):
        """
        Cross-attention: given q and kv, return new encoding of q
        q input shape: (batch, seq, q_dim), output: (batch, seq, dim)
        """
        super().__init__()
        self.heads = heads
        self.scale = kv_dim ** -0.5

        self.to_q = nn.Linear(q_dim, dim, bias=True)
        self.to_k = nn.Linear(kv_dim, dim, bias=True)
        self.to_v = nn.Linear(kv_dim, dim, bias=True)

        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(droupout)

    def forward(self, x, y, mask=None):
        # x shape: (batch, seq1, q_dim)
        # y shape: (batch, seq2, kv_dim)
        b, n, _, h = *x.shape, self.heads
        by, ny, _, hy = *y.shape, self.heads
        assert b == by

        qheads, kheads, vheads = self.to_q(x), self.to_k(y), self.to_v(y)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (qheads, kheads, vheads))
        
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_heads, dropout):
        super().__init__()
        inner_dim = dim_heads * heads
        self.heads = heads
        self.scale = dim_heads ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dim_heads=dim_heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x


class CrossTransformer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, drouput):
        """
        Cross-transformer for bidirectional feature exchange
        Input x, y with same dim; outputs new representations of both
        """
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.norm_x2 = nn.LayerNorm(dim)
        self.norm_y2 = nn.LayerNorm(dim)

        self.cross_attention_x = CrossAttention(dim, dim, dim, heads=heads, droupout=drouput)
        self.cross_attention_y = CrossAttention(dim, dim, dim, heads=heads, droupout=drouput)

        self.mlp_x = MLP_Block(dim, mlp_dim, dropout=drouput)
        self.mlp_y = MLP_Block(dim, mlp_dim, dropout=drouput)

    def forward(self, x, y, mask=None):
        assert mask is None
        shortcut_x = x
        shortcut_y = y

        x = self.norm_x(x)
        y = self.norm_y(y)

        x = self.cross_attention_x(x, y, mask=mask)
        y = self.cross_attention_y(y, x, mask=mask)

        x = shortcut_x + x
        y = shortcut_y + y

        x = x + self.mlp_x(self.norm_x2(x))
        y = y + self.mlp_y(self.norm_y2(y))

        return x, y


class HSINet(nn.Module):
    """
    GGCL Hyperspectral Image Classification Network
    Based on Vision Transformer with contrastive learning support
    """
    def __init__(self, params):
        super(HSINet, self).__init__()
        self.params = params
        net_params = params['net']
        data_params = params['data']

        num_classes = data_params.get("num_classes", 16)
        patch_size = data_params.get("patch_size", 13)
        self.spectral_size = data_params.get("spectral_size", 144)

        depth = net_params.get("depth", 1)
        heads = net_params.get("heads", 8)
        mlp_dim = net_params.get("mlp_dim", 8)
        dropout = net_params.get("dropout", 0)
        conv2d_out = 64
        dim = net_params.get("dim", 64)
        dim_heads = dim
        mlp_head_dim = dim

        image_size = patch_size * patch_size

        self.pixel_patch_embedding = nn.Linear(self.spectral_size, dim)

        self.local_trans_pixel = Transformer(
            dim=dim, depth=depth, heads=heads, 
            dim_heads=dim_heads, mlp_dim=mlp_dim, dropout=dropout
        )
        self.new_image_size = image_size
        self.pixel_pos_embedding = nn.Parameter(torch.randn(1, self.new_image_size + 1, dim))
        self.pixel_pos_scale = nn.Parameter(torch.ones(1) * 0.01)

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=self.spectral_size, out_channels=conv2d_out, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(conv2d_out),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv2d_out, out_channels=dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.cls_token_pixel = nn.Parameter(torch.randn(1, 1, dim))
        self.to_latent_pixel = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.mlp_head.weight)
        torch.nn.init.normal_(self.mlp_head.bias, std=1e-6)
        self.dropout = nn.Dropout(0.1)

        mlp_dim1 = mlp_head_dim
        mlp_dim2 = mlp_dim1 * 2
        self.classifier_mlp = nn.Sequential(
            nn.Linear(mlp_dim1, mlp_dim2),
            nn.BatchNorm1d(mlp_dim2),
            nn.ReLU(),
            nn.Linear(mlp_dim2, mlp_dim2),
            nn.BatchNorm1d(mlp_dim2),
            nn.ReLU(),
            nn.Linear(mlp_dim2, num_classes),
        )

    def classifier(self, x):
        """Detached classifier for pseudo-labeling"""
        x = x.detach()
        return self.classifier_mlp(x)

    def encoder_block(self, x):
        """
        Encode input patch
        x: (batch, spectral, width, height)
        """
        x_pixel = x
        b, s, w, h = x_pixel.shape
        
        x_pixel = self.conv2d_features(x_pixel)
        x_pixel = rearrange(x_pixel, 'b s w h -> b (w h) s')

        cls_tokens_pixel = self.cls_token_pixel.expand(x_pixel.shape[0], -1, -1)
        x_pixel = torch.cat((cls_tokens_pixel, x_pixel), dim=1)
        x_pixel = x_pixel + self.pixel_pos_embedding[:, :] * self.pixel_pos_scale

        x_pixel = self.local_trans_pixel(x_pixel)

        logit_pixel = self.to_latent_pixel(x_pixel[:, 0])
        reduce_x = torch.mean(x_pixel, dim=1)

        return logit_pixel, reduce_x

    def forward(self, x, left=None, right=None):
        """
        Forward pass
        x: (batch, spectral, width, height)
        left, right: augmented views for contrastive learning (optional)
        
        Returns:
            logits: class predictions
            mean_left: mean features from left augmentation
            mean_right: mean features from right augmentation
        """
        logit_x, _ = self.encoder_block(x)
        mean_left, mean_right = None, None
        
        if left is not None and right is not None:
            _, mean_left = self.encoder_block(left)
            _, mean_right = self.encoder_block(right)

        return self.mlp_head(logit_x), mean_left, mean_right
