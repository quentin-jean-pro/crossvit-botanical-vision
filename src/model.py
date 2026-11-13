"""
CrossViT architecture implementation in PyTorch.
Supports multi-scale dual-branch tokenization and cross-attention fusion.
Based on Chen et al. (ICCV 2021) with custom attention rollout tracking for interpretability.
"""

import types
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Model Registry ---
_MODEL_REGISTRY = {}

def register_model(fn):
    _MODEL_REGISTRY[fn.__name__] = fn
    return fn

# Model weights URL (official public IBM CrossViT checkpoint)
_MODEL_URLS = {
    'crossvit_tiny_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_tiny_224.pth',
}

def to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Simple truncated normal initialization
    with torch.no_grad():
        nn.init.normal_(tensor, mean=mean, std=std)
        tensor.clamp_(min=a*std + mean, max=b*std + mean)
    return tensor


class DropPath(nn.Module):
    """Stochastic depth per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """Multi-Layer Perceptron used in Transformer blocks."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """Splits image into non-overlapping patches and projects to embedding space."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, 7, 4, 3), nn.ReLU(True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 3, 0), nn.ReLU(True),
                    nn.Conv2d(embed_dim // 2, embed_dim, 3, 1, 1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, 7, 4, 3), nn.ReLU(True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 2, 1), nn.ReLU(True),
                    nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Standard multi-head self-attention with rollout caching."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Cached attention map for visualization and IoU loss
        self.last_attn = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        self.last_attn = attn.detach()

        x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Standard Vision Transformer Block."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    """Cross-attention mechanism exchanging CLS tokens between branches."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Store cross-attention weights for interpretability and loss terms
        self.last_attention = None

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn.softmax(dim=-1)

        self.last_attention = attn_weights.detach().clone()

        attn_dropped = self.attn_drop(attn_weights)
        x = (attn_dropped @ v).transpose(1, 2).reshape(B, 1, C)
        return self.proj_drop(self.proj(x))


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with layer normalization and residual connection."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MultiScaleBlock(nn.Module):
    """Dual-branch multi-scale transformer block with cross-attention fusion."""
    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        num_branches = len(dim)
        self.num_branches = num_branches

        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = [
                Block(dim[d], num_heads[d], mlp_ratio[d], qkv_bias, qk_scale, drop, attn_drop,
                      drop_path[i] if isinstance(drop_path, list) else drop_path,
                      act_layer=act_layer, norm_layer=norm_layer)
                for i in range(depth[d])
            ]
            self.blocks.append(nn.ModuleList(tmp))

        if depth[0] > 0 and num_branches > 1:
            self.projs = nn.ModuleList([
                nn.Sequential(norm_layer(dim[i]), act_layer(), nn.Linear(dim[i], dim[(i + 1) % num_branches]))
                for i in range(num_branches)
            ])
            self.fusion = nn.ModuleList([
                CrossAttentionBlock(
                    dim[(i + 1) % num_branches],
                    num_heads[(i + 1) % num_branches],
                    mlp_ratio[i],
                    qkv_bias,
                    qk_scale,
                    drop,
                    attn_drop,
                    drop_path[-1] if isinstance(drop_path, list) else drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for i in range(num_branches)
            ])
            self.revert_projs = nn.ModuleList([
                nn.Sequential(norm_layer(dim[(i + 1) % num_branches]), act_layer(), nn.Linear(dim[(i + 1) % num_branches], dim[i]))
                for i in range(num_branches)
            ])

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        for d in range(self.num_branches):
            for blk in self.blocks[d]:
                xs[d] = blk(xs[d])

        if len(self.blocks[0]) == 0 or self.num_branches == 1:
            return xs

        # Cross-branch token exchange
        cls_tokens = [x[:, 0:1] for x in xs]
        patch_tokens = [x[:, 1:] for x in xs]

        cls_projected = [self.projs[i](cls_tokens[i]) for i in range(self.num_branches)]
        cross_inputs = [
            torch.cat((cls_projected[i], patch_tokens[(i + 1) % self.num_branches]), dim=1)
            for i in range(self.num_branches)
        ]

        cls_fused = [self.fusion[i](cross_inputs[i]) for i in range(self.num_branches)]
        cls_restored = [self.revert_projs[i](cls_fused[i]) for i in range(self.num_branches)]

        return [torch.cat((cls_restored[i], patch_tokens[i]), dim=1) for i in range(self.num_branches)]


class VisionTransformer(nn.Module):
    """Multi-scale Vision Transformer with dual branches."""
    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=1000,
                 embed_dim=(192, 384), depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, multi_conv=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
        self.num_branches = len(embed_dim)

        self.patch_embed = nn.ModuleList([
            PatchEmbed(self.img_size[i], patch_size[i], in_chans, embed_dim[i], multi_conv)
            for i in range(self.num_branches)
        ])
        num_patches = [pe.num_patches for pe in self.patch_embed]

        self.cls_token = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)
        ])
        self.pos_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_patches[i] + 1, embed_dim[i])) for i in range(self.num_branches)
        ])
        self.pos_drop = nn.Dropout(p=drop_rate)

        total_depth = sum([sum(x) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dpr_ptr = 0

        self.blocks = nn.ModuleList()
        for idx in range(len(depth)):
            curr = sum(depth[idx])
            self.blocks.append(MultiScaleBlock(
                embed_dim, num_patches, depth[idx], num_heads, mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[dpr_ptr:dpr_ptr + curr], norm_layer=norm_layer
            ))
            dpr_ptr += curr

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        self.head = nn.ModuleList([
            nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity()
            for i in range(self.num_branches)
        ])

        for i in range(self.num_branches):
            trunc_normal_(self.pos_embed[i], std=.02)
            trunc_normal_(self.cls_token[i], std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B, xs = x.shape[0], []
        for i in range(self.num_branches):
            x_ = F.interpolate(x, self.img_size[i], mode='bicubic') if x.shape[-1] != self.img_size[i] else x
            tmp = self.patch_embed[i](x_)
            cls = self.cls_token[i].expand(B, -1, -1)
            tmp = self.pos_drop(torch.cat((cls, tmp), dim=1) + self.pos_embed[i])
            xs.append(tmp)
        for blk in self.blocks:
            xs = blk(xs)
        return [self.norm[i](x) for i, x in enumerate(xs)]

    def forward(self, x):
        xs = self.forward_features(x)
        return torch.mean(torch.stack([self.head[i](x[:, 0]) for i, x in enumerate(xs)]), dim=0)


def dual_forward_impl(self, x1, x2):
    """
    Forward pass receiving two distinct images for the two branches.
    Allows feeding raw and segmented images simultaneously to compare multiscale interactions.
    """
    B, xs = x1.shape[0], []
    for i, x_in in enumerate([x1, x2]):
        if x_in.shape[-1] != self.img_size[i]:
            x_in = F.interpolate(x_in, self.img_size[i], mode='bicubic')
        tmp = self.patch_embed[i](x_in)
        cls = self.cls_token[i].expand(B, -1, -1)
        tmp = self.pos_drop(torch.cat((cls, tmp), dim=1) + self.pos_embed[i])
        xs.append(tmp)
    for blk in self.blocks:
        xs = blk(xs)
    xs = [self.norm[i](x) for i, x in enumerate(xs)]
    return torch.mean(torch.stack([self.head[i](x[:, 0]) for i, x in enumerate(xs)]), dim=0)


@register_model
def crossvit_tiny_224(pretrained=False, num_classes=2, **kwargs):
    """
    Original CrossViT-Tiny: Small branch (patch 12) + Large branch (patch 16).
    Evaluated across input combinations (Raw, Segmented, Cross).
    """
    model = VisionTransformer(
        img_size=[240, 224], patch_size=[12, 16], embed_dim=[96, 192],
        depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]], num_heads=[3, 3],
        mlp_ratio=[4, 4, 1], qkv_bias=True, num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model.dual_forward = types.MethodType(dual_forward_impl, model)

    if pretrained:
        try:
            ckpt = torch.hub.load_state_dict_from_url(
                _MODEL_URLS['crossvit_tiny_224'], map_location="cpu", check_hash=True
            )
            sd = ckpt.get('model', ckpt)
            msd = model.state_dict()
            filtered = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
            msd.update(filtered)
            model.load_state_dict(msd)
        except Exception:
            # Fall back to random initialization if offline
            pass
    return model


@register_model
def crossvit_tiny_same_res(pretrained=False, num_classes=2, **kwargs):
    """
    Same-resolution control variant: both branches use patch 16.
    Isolates the mathematical contribution of multiscale tokenization.
    """
    model = VisionTransformer(
        img_size=[224, 224], patch_size=[16, 16], embed_dim=[192, 192],
        depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]], num_heads=[3, 3],
        mlp_ratio=[4, 4, 1], qkv_bias=True, num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model.dual_forward = types.MethodType(dual_forward_impl, model)
    return model
