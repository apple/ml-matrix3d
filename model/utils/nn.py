#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from typing import Optional
import math
import torch
import torch.nn as nn

import xformers.ops as xops
from diffusers.models.attention_processor import Attention

from .pos_encoder import FeaturePositionalEncoding


def modulate(x, shift, scale):
    if x.is_nested:
        return x * (1 + scale) + shift
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def convert_tensor_to_nested_tensor(tensor_list, in_nt_tensor):
    '''convert tensor to nested tensor'''
    batch_size = in_nt_tensor.size(0)
    out_nt_tensor = []
    for tensor in tensor_list:
        nt_tensor = torch.nested.as_nested_tensor([tensor[i].unsqueeze(0).repeat(in_nt_tensor[i].shape[0], 1) for i in range(batch_size)])
        out_nt_tensor.append(nt_tensor)
        
    return out_nt_tensor


def restore_nested_tensor_to_tensor(nt_tensor, orig_shape, mask, value=0.):
    restored_tensor = torch.ones(orig_shape, dtype=nt_tensor.dtype, device=nt_tensor.device) * value
    for i, m in enumerate(mask):
        restored_tensor[i][m] = nt_tensor[i]

    return restored_tensor


def full_to_packed(data, mask):
    seqlist = [seq[m] for seq, m in zip(data, mask)]
    seqlen = [seq.shape[0] for seq in seqlist]
    packed = torch.cat(seqlist, dim=0).unsqueeze(0)
    return packed, seqlen


def packed_to_nested(data, seqlen):
    data = torch.nested.as_nested_tensor(list(data[0].split(seqlen, dim=0)))
    return data


def nested_to_packed(data):
    seqlist = [seq for seq in data]
    seqlen = [seq.shape[0] for seq in seqlist]
    packed = torch.cat(seqlist, dim=0).unsqueeze(0)
    return packed, seqlen


def packed_to_padded(data, seqlen, total=None, fill=0.):
    if total is None:
        total = max(seqlen)
    return torch.stack([
        torch.cat([
            seq,
            torch.full((total-seq.shape[0], *seq.shape[1:]), fill, dtype=seq.dtype, device=seq.device)
        ], dim=0)
        for seq in data[0].split(seqlen, dim=0)
    ], dim=0), seqlen


def padded_to_packed(data, seqlen):
    return torch.cat([
        seq[:l] for seq, l in zip(data, seqlen)
    ], dim=0).unsqueeze(0), seqlen


def packed_to_full(data, mask, fill=0.):
    out = torch.full((*mask.shape, *data.shape[2:]), fill, dtype=data.dtype, device=data.device)
    out[mask] = data[0]
    return out, mask


def full_to_padded(data, mask, total=None, fill=0.):
    return packed_to_padded(*full_to_packed(data, mask), total=total, fill=fill)


def padded_to_full(data, seqlen, mask, fill=0.):
    return packed_to_full(padded_to_packed(data, seqlen)[0], mask, fill=fill)


# modified from https://github.com/meta-llama/llama/blob/main/llama/model.py
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        nested = False
        if x.is_nested:  # NOTE assume seq dim is 1
            nested = True
            x, seqlen = nested_to_packed(x)
        x = self._norm(x) * (1 + self.weight)
        if nested:
            x = packed_to_nested(x, seqlen)
        return x


class HolisticAttnProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_pos_s=None, key_pos_s=None, query_pos_r=None, key_pos_r=None, seqlen_q=None, seqlen_kv=None,
    ) -> torch.Tensor:
        query = hidden_states
        key = value = encoder_hidden_states

        if query_pos_s is not None:
            query = query + query_pos_s
        if key_pos_s is not None:
            key = key + key_pos_s

        query, _ = padded_to_packed(query, seqlen_q)
        key, _ = padded_to_packed(key, seqlen_kv)
        value, _ = padded_to_packed(value, seqlen_kv)

        q = attn.to_q(query)
        k = attn.to_k(key)
        v = attn.to_v(value)

        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        inner_dim = k.shape[-1]
        head_dim = inner_dim // attn.heads

        q = q.reshape(1, -1, attn.heads, head_dim)  # batch_size=1 because it's packed
        k = k.reshape(1, -1, attn.heads, head_dim)
        v = v.reshape(1, -1, attn.heads, head_dim)

        q = attn.norm_q(q).to(q.dtype)
        k = attn.norm_k(k).to(k.dtype)

        if query_pos_r is not None:
            query_pos_r, _ = padded_to_packed(query_pos_r, seqlen_q)
            q = FeaturePositionalEncoding.apply_rotary_emb(q, query_pos_r)
        if key_pos_r is not None:
            key_pos_r, _ = padded_to_packed(key_pos_r, seqlen_kv)
            k = FeaturePositionalEncoding.apply_rotary_emb(k, key_pos_r)

        x = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(seqlen_q, seqlen_kv), 
            p=attn.attn_drop.p if attn.training and hasattr(attn, 'attn_drop') else 0.,
        )
        x = x.reshape(1, -1, inner_dim)
        x = attn.to_out[0](x)  # linear
        x = attn.to_out[1](x)  # dropout

        x, _ = packed_to_padded(x, seqlen_q, max(seqlen_q))
        
        return x



class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def initialize_weights(self):
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MultiLayerPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        assert patch_size in [2, 4, 6, 8, 16]
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        n_down = round(math.log2(patch_size))
        self.proj = [nn.Conv2d(in_chans, embed_dim, 3, 1, 1)]
        for i in range(n_down):
            self.proj.append(nn.SiLU(inplace=True))
            self.proj.append(nn.Conv2d(embed_dim, embed_dim, 2, 2, 0))
            self.proj.append(nn.SiLU(inplace=True))
            self.proj.append(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1))
        self.proj = nn.Sequential(*self.proj)

    def initialize_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Conv2d):
                w = m.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        x: [N, C, H, W]
        out: [N, C, H, W]
        '''
        return self.proj(x)

class MultiLayerFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        assert patch_size in [2, 4, 6, 8, 16]
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        n_up = round(math.log2(patch_size))
        self.proj = []
        for i in range(n_up - 1):
            self.proj.append(nn.ConvTranspose2d(hidden_size, hidden_size, 2, 2, 0, 0))
            self.proj.append(nn.SiLU(inplace=True))
            self.proj.append(nn.Conv2d(hidden_size, hidden_size, 3, 1, 1))
            self.proj.append(nn.SiLU(inplace=True))
        self.proj.append(nn.ConvTranspose2d(hidden_size, out_channels, 2, 2, 0, 0))
        self.proj = nn.Sequential(*self.proj)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def initialize_weights(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        for m in self.proj.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                w = m.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.proj[-1].weight, 0)

    def forward(self, x, c):
        '''
        x: [B, N, H, W, D]
        c: [B, D]
        out: [B, N, D, H, W]
        '''
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        seq_dims = x.shape[1:4]
        x = x.flatten(1, 3)  # [B, L, D]
        x = modulate(self.norm_final(x), shift, scale)
        x = x.unflatten(1, seq_dims)  # [B, N, H, W, D]
        x = x.permute(0, 1, 4, 2, 3)  # [B, N, D, H, W]
        batch_dims = x.shape[0:2]
        x = x.flatten(0, 1)  # [BN, D, H, W]
        x = self.proj(x)
        return x.unflatten(0, batch_dims)
