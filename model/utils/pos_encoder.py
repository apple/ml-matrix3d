#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from typing import Union, Tuple
import numpy as np
import torch
import torch.nn as nn


class FeaturePositionalEncoding(nn.Module):
    mod_ind_table = {
        'rgb': 0,
        'ray': 1,
        'depth': 2,
        'local_caption': 3,
        'global_caption': 4,
    }

    @classmethod
    def _get_sinusoid_encoding_table(cls, n_position, d_hid, base):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(base, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    @classmethod
    def _get_2d_sincos_pos_embed(cls, grid_size, embed_dim, base, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0]/base_size) / lewei_scale
        grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1]/base_size) / lewei_scale
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

        pos_embed = cls._get_2d_sincos_pos_embed_from_grid(embed_dim, grid, base)
        if cls_token and extra_tokens > 0:
            pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
        return torch.FloatTensor(pos_embed).unsqueeze(0)

    @classmethod
    def _get_2d_sincos_pos_embed_from_grid(cls, embed_dim, grid, base):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = cls._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], base)  # (H*W, D/2)
        emb_w = cls._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], base)  # (H*W, D/2)

        return np.stack([emb_h, emb_w], axis=2).reshape((-1, embed_dim))  # transpose to h,w,h,w,...

    @classmethod
    def _get_1d_sincos_pos_embed_from_grid(cls, embed_dim, pos, base):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / base ** omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        return np.stack([emb_sin, emb_cos], axis=2).reshape((-1, embed_dim))  # transpose to sin,cos,sin,cos,...

    @classmethod
    def _get_2d_rotary_pos_embed(cls, grid_size, embed_dim, crops_coords=None, use_real=True, base_size=16):
        """
        RoPE for image tokens with 2d structure.

        Args:
        embed_dim: (`int`):
            The embedding dimension size
        crops_coords (`Tuple[int]`)
            The top-left and bottom-right coordinates of the crop.
        grid_size (`Tuple[int]`):
            The grid size of the positional embedding.
        use_real (`bool`):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.

        Returns:
            `torch.Tensor`: positional embdding with shape `( grid_size * grid_size, embed_dim/2)`.
        """
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        if crops_coords is None:
            crops_coords = ((0,0), grid_size)
        start, stop = crops_coords
        grid_h = np.linspace(start[0], stop[0], grid_size[0], endpoint=False, dtype=np.float32) / (grid_size[0]/base_size)
        grid_w = np.linspace(start[1], stop[1], grid_size[1], endpoint=False, dtype=np.float32) / (grid_size[1]/base_size)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)  # [2, W, H]

        grid = grid.reshape([2, 1, *grid.shape[1:]])
        pos_embed = cls._get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=use_real)
        return torch.cat(pos_embed, dim=-1).unsqueeze(0)

    @classmethod
    def _get_2d_rotary_pos_embed_from_grid(cls, embed_dim, grid, use_real=False):
        assert embed_dim % 4 == 0

        # use half of dimensions to encode grid_h
        emb_h = cls._get_1d_rotary_pos_embed(embed_dim // 2, grid[0].reshape(-1), use_real=use_real)  # (H*W, D/4)
        emb_w = cls._get_1d_rotary_pos_embed(embed_dim // 2, grid[1].reshape(-1), use_real=use_real)  # (H*W, D/4)

        if use_real:
            cos = torch.cat([emb_h[0], emb_w[0]], dim=1)  # (H*W, D/2)
            sin = torch.cat([emb_h[1], emb_w[1]], dim=1)  # (H*W, D/2)
            return cos, sin
        else:
            emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D/2)
            return emb

    @classmethod
    def _get_1d_rotary_pos_embed(cls, dim: int, pos: Union[np.ndarray, int], theta: float = 10000.0, use_real=False):
        """
        Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

        This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
        index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
        data type.

        Args:
            dim (`int`): Dimension of the frequency tensor.
            pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
            theta (`float`, *optional*, defaults to 10000.0):
                Scaling factor for frequency computation. Defaults to 10000.0.
            use_real (`bool`, *optional*):
                If True, return real part and imaginary part separately. Otherwise, return complex numbers.

        Returns:
            `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
        """
        if isinstance(pos, int):
            pos = np.arange(pos)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [D/2]
        t = torch.from_numpy(pos).to(freqs.device)  # type: ignore  # [S]
        freqs = torch.outer(t, freqs).float()  # type: ignore   # [S, D/2]
        if use_real:
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
            return freqs_cos, freqs_sin
        else:
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
            return freqs_cis

    @classmethod
    def apply_rotary_emb(cls,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
        to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
        reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
        tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`):
                Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. [1, S, 2*D]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        cos, sin = freqs_cis.unsqueeze(-2).to(x.device).unflatten(-1, (2, -1)).unbind(-2)  # [1, S, 1, D]
        x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out

    def __init__(self, pe_config, feature_dim=1152, num_heads=16, grid_size=64):
        super().__init__()
        self.pe_config = pe_config
        self.feature_dim = feature_dim
        self.grid_size = grid_size  # this denotes the dense grid size

        if pe_config['view']['type'] in ['sinusoid', 'sinusoid_all']:
            self.register_buffer(
                "view_pos_table",
                self._get_sinusoid_encoding_table(
                    pe_config['view']['max'], feature_dim, pe_config['view']['base']
                ), 
            )
            # self.view_pos_enc_map = nn.Linear(feature_dim, feature_dim)  # TODO add a field in config to control
            # nn.init.zeros_(self.view_pos_enc_map.weight)
            # nn.init.zeros_(self.view_pos_enc_map.bias)
        elif pe_config['view']['type'] == 'rope':
            raise NotImplementedError

        if pe_config['pos']['type'] in ['sinusoid', 'sinusoid_all']:
            self.register_buffer(
                "token_pos_table",
                self._get_2d_sincos_pos_embed(
                    grid_size, feature_dim, pe_config['pos']['base'], base_size=pe_config['pos']['base_size']
                ), 
            )
        elif pe_config['pos']['type'] == 'rope':
            self.register_buffer(
                "token_pos_table",
                self._get_2d_rotary_pos_embed(
                    grid_size, feature_dim // num_heads, base_size=pe_config['pos']['base_size']
                ), 
            )

        if pe_config['mod']['max'] > 1:
            if pe_config['mod']['type'] in ['sinusoid', 'sinusoid_all']:
                self.register_buffer(
                    "mod_pos_table",
                    self._get_sinusoid_encoding_table(
                        pe_config['mod']['max'], feature_dim, pe_config['mod']['base']
                    ),
                )
                # self.mod_pos_enc_map = nn.Linear(feature_dim, feature_dim)
                # nn.init.zeros_(self.mod_pos_enc_map.weight)
                # nn.init.zeros_(self.mod_pos_enc_map.bias)
            elif pe_config['mod']['type'] == 'rope':
                raise NotImplementedError
        
        
    def forward(self, x, mod_ind, view_ids=None, zero_view_pe=False, zero_patch_pe=False):
        ''' view_ids: (B, N)
        '''
        if len(x.shape) == 5:
            batch_size, num_images, H, W, feat_dim = x.shape
            num_patches = H * W
            x = x.reshape(batch_size, num_images, num_patches, feat_dim)
        elif len(x.shape) == 4:
            batch_size, num_images, num_patches, feat_dim = x.shape
        else:
            raise ValueError('Invalid input shape. Expected (B, N, H, W, D) or (B, N, L, D)')
        # To encode view index

        pe_dict = {
            'rope': None,
            'sinusoid': 0,
            'sinusoid_all': 0,
        }

        # To encode view index
        if self.pe_config['view']['type'] in ['sinusoid', 'sinusoid_all']:
            if zero_view_pe:
                pe1 = 0.0
            else:
                if view_ids is not None:
                    pe1 = self.view_pos_table[:, view_ids.reshape(-1), :feat_dim].clone().detach()
                    pe1 = pe1.reshape((batch_size, num_images, 1, feat_dim))
                    pe1 = pe1.repeat((1, 1, num_patches, 1))
                else:
                    pe1 = self.view_pos_table[:, :num_images, :feat_dim].clone().detach()
                    pe1 = pe1.reshape((1, num_images, 1, feat_dim))
                    pe1 = pe1.repeat((batch_size, 1, num_patches, 1))
                # pe1 = self.view_pos_enc_map(pe1)
            pe_dict[ self.pe_config['view']['type'] ] = pe_dict[ self.pe_config['view']['type'] ] + pe1
        elif self.pe_config['view']['type'] == 'rope':
            raise NotImplementedError

        # To encode patch index
        if self.pe_config['pos']['type'] in ['sinusoid', 'sinusoid_all']:
            if zero_patch_pe:
                pe2 = 0.0
            else:
                pe2 = self.token_pos_table[:, :num_patches, :feat_dim].clone().detach()
                pe2 = pe2.reshape((1, 1, num_patches, feat_dim))
                pe2 = pe2.repeat((batch_size, num_images, 1, 1))
            pe_dict[ self.pe_config['pos']['type'] ] = pe_dict[ self.pe_config['pos']['type'] ] + pe2
        elif self.pe_config['pos']['type'] == 'rope':
            if zero_patch_pe:  # identity rope
                rope_dim = self.token_pos_table.shape[-1]
                pe2 = torch.cat([
                    torch.ones(1, num_patches, rope_dim//2, dtype=self.token_pos_table.dtype, device=self.token_pos_table.device),
                    torch.zeros(1, num_patches, rope_dim//2, dtype=self.token_pos_table.dtype, device=self.token_pos_table.device),
                ], dim=-1)
            else:
                pe2 = self.token_pos_table[:, :num_patches, :].clone().detach()
            pe2 = pe2.reshape((1, 1, num_patches, -1))
            pe2 = pe2.repeat((batch_size, num_images, 1, 1))
            pe_dict[ self.pe_config['pos']['type'] ] = pe2

        # To encode modality index
        if self.pe_config['mod']['max'] > 1:
            if self.pe_config['mod']['type'] in ['sinusoid', 'sinusoid_all']:
                pe3 = self.mod_pos_table[:, mod_ind, :feat_dim].clone().detach()
                pe3 = pe3.reshape((1, 1, 1, feat_dim))
                pe3 = pe3.repeat((batch_size, num_images, num_patches, 1))
                # pe3 = self.mod_pos_enc_map(pe3)
                pe_dict[ self.pe_config['mod']['type'] ] = pe_dict[ self.pe_config['mod']['type'] ] + pe3
            elif self.pe_config['mod']['type'] == 'rope':
                raise NotImplementedError

        x = x + pe_dict.pop('sinusoid')
        pe_dict = {k: v if isinstance(v, torch.Tensor) else None for k, v in pe_dict.items()}
        return x, pe_dict
