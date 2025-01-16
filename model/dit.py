#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

# Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Mlp, PatchEmbed
from .hunyuan import HunyuanDiT2DModel
from .utils.nn import (
    RMSNorm,
    HolisticAttnProcessor,
    modulate,
    full_to_padded,
    MultiLayerPatchEmbed,
)
from .utils.pos_encoder import FeaturePositionalEncoding
from diffusers.models.embeddings import (
    HunyuanCombinedTimestepTextSizeStyleEmbedding,
    PixArtAlphaTextProjection,
)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype=next(self.mlp.parameters()).dtype)
        t_emb = self.mlp(t_freq) 
        return t_emb


class AttentionNestedTensor(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            kdim: int = None,
            vdim: int = None,
            bias: bool = False,
            qk_norm: str = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.heads = num_heads
        self.head_dim = embed_dim // num_heads
        # self.scale = self.head_dim ** -0.5
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_k = nn.Linear(embed_dim, self.kdim, bias=bias)
        self.to_v = nn.Linear(embed_dim, self.kdim, bias=bias)

        if qk_norm is not None:
            if qk_norm == "layernorm":
                self.norm_q = nn.LayerNorm(self.head_dim, eps=1e-6)
                self.norm_k = nn.LayerNorm(self.head_dim, eps=1e-6)
            elif qk_norm == "rmsnorm":
                self.norm_q = RMSNorm(self.head_dim, eps=1e-6)
                self.norm_k = RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.norm_q, self.norm_k = nn.Identity(), nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.ModuleList([nn.Linear(embed_dim, embed_dim), nn.Dropout(proj_drop)])

        self.attn_processor = HolisticAttnProcessor()

    def forward(self, query, key_value, query_pos_s=None, key_pos_s=None, query_pos_r=None, key_pos_r=None, seqlen_q=None, seqlen_kv=None) -> torch.Tensor:
        return self.attn_processor(
            self, query, key_value, None, query_pos_s, key_pos_s, query_pos_r, key_pos_r, seqlen_q, seqlen_kv
        )


class DiTBlockSelfAttn(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        qk_norm=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = AttentionNestedTensor(
            embed_dim=hidden_size, num_heads=num_heads, bias=True, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, pe_x_s=None, pe_x_r=None, seqlen_x=None):
        ''' x could be nested tensor'''
        (
            shift_msa, scale_msa, gate_msa,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        # self-attn + mlp
        before_sa = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            before_sa, before_sa, pe_x_s, pe_x_s, pe_x_r, pe_x_r, seqlen_x, seqlen_x)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))
                
        return x
    

class DiTBlockCrossAttn(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning and cross attention.
    """

    def __init__(
        self,
        hidden_size,
        encoding_size,
        num_heads,
        mlp_ratio=4.0,
        qk_norm=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = AttentionNestedTensor(
            embed_dim=hidden_size, num_heads=num_heads, kdim=encoding_size, vdim=encoding_size, bias=True, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = AttentionNestedTensor(
            embed_dim=hidden_size, num_heads=num_heads, bias=True, qk_norm=qk_norm)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        # )
        self.scale_shift_table = nn.Parameter(torch.randn(9, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, enc, c, pe_x_s=None, pe_enc_s=None, pe_x_r=None, pe_enc_r=None, seqlen_x=None, seqlen_enc=None, uncond=False):
        ''' x, enc could be nested tensor'''
        (
            shift_msa, scale_msa, gate_msa,
            shift_mca, scale_mca, gate_mca,
            shift_mlp, scale_mlp, gate_mlp,
        ) = (self.scale_shift_table.reshape(1,-1) + c).chunk(9, dim=1)
        # cross-attn + self-attn + mlp
        before_sa = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            before_sa, before_sa, before_sa, pe_x_s, pe_x_s, pe_x_r, pe_x_r, seqlen_x, seqlen_x)
        if isinstance(uncond, bool):
            uncond = torch.full((x.shape[0],), uncond, dtype=torch.bool, device=x.device)
        x = x + (~uncond).to(x.dtype)[:, None, None] * gate_mca.unsqueeze(1) * self.cross_attn(
            modulate(self.norm2(x), shift_mca, scale_mca), enc, enc, pe_x_s, pe_enc_s, pe_x_r, pe_enc_r, seqlen_x, seqlen_enc)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp))

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_size=1152,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        use_mask=False,
        modalities=None,
        pe_config=None,
        qk_norm=None,
        cross_attention_dim: int = 1024,
        cross_attention_dim_t5: int = 2048,
        pooled_projection_dim: int = 1024,
        text_len: int = 77,
        text_len_t5: int = 256,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.depth = depth
        self.use_mask = use_mask
        self.modalities = modalities
        self.training = True
        self.pe_config = pe_config
        self.qk_norm = qk_norm

        # grid size of all 2D modalities should be identical
        grid_sizes = [v['width'] // v['patch_size'] for k, v in modalities.items() if v['dimensions'] == 2]
        assert all([grid_sizes[0] == it for it in grid_sizes])
        grid_size = grid_sizes[0]
        # Init PatchEmbedders
        # PE (view_id, token pos, modality type)
        self.pos_enc = FeaturePositionalEncoding(pe_config, hidden_size, num_heads, grid_size)

        # image-like modality-specific embedders
        self.embedders = {}
        has_text_input = False
        for mod, mod_conf in modalities.items():
            if mod_conf['dimensions'] != 2: 
                has_text_input = True
                continue
            if mod == 'depth' and mod_conf['patch_size'] > 4:
                self.embedders[mod] = MultiLayerPatchEmbed(
                    img_size=mod_conf['width'],
                    patch_size=mod_conf['patch_size'],
                    in_chans=sum(mod_conf['cond_channel'].values()),
                    embed_dim=hidden_size,
                )
            else:
                self.embedders[mod] = PatchEmbed(
                    img_size=mod_conf['width'],
                    patch_size=mod_conf['patch_size'],
                    in_chans=sum(mod_conf['cond_channel'].values()),
                    embed_dim=hidden_size,
                    bias=True,
                    flatten=False,
                )
        self.embedders = nn.ModuleDict(self.embedders)
        # text-like modality-specific embedders
        if has_text_input:
            text_embed_dim = [sum(mod_conf['cond_channel'].values()) for mod, mod_conf in modalities.items() if mod_conf['dimensions'] == 1]
            assert all([text_embed_dim[0] == i for i in text_embed_dim]), 'all text modalities must have the same embedding dim'
            text_embed_dim = text_embed_dim[0]
            # self.cond_text_embedder = PixArtAlphaTextProjection(
            #     in_features=text_embed_dim,
            #     hidden_size=text_embed_dim,
            #     out_features=hidden_size,
            #     act_fn="silu_fp32",
            # )
            self.cond_text_embedder = nn.Identity()
            self.cond_text_embedder_combined = None  # to be assigned after creation

        # modality-specific normalization layer
        self.mod_norm = nn.ModuleDict({mod: nn.LayerNorm(hidden_size) for mod, mod_conf in modalities.items() if mod_conf['dimensions'] == 2})
        if any([key in modalities for key in ['local_caption', 'global_caption']]):
            self.mod_norm['caption'] = nn.LayerNorm(hidden_size)

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.mv_encoder = nn.ModuleList(
            [
                DiTBlockSelfAttn(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(depth)
            ]
        )

        self.grad_checkpoint = False

        # Init weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        for modality in self.modalities.keys():
            if self.modalities[modality]['dimensions'] != 2: continue
            embedder = self.embedders[modality]
            if isinstance(embedder, MultiLayerPatchEmbed):
                embedder.initialize_weights()
            else:
                w = embedder.proj.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                nn.init.constant_(embedder.proj.bias, 0)
        # init text embedder
        # if any([key in self.modalities for key in ['local_caption', 'global_caption']]):
        #     for m in [self.cond_text_embedder.linear_1, self.cond_text_embedder.linear_2]:
        #         w = m.weight.data
        #         nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        #         nn.init.constant_(m.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.mv_encoder:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.grad_checkpoint = True

    def caption_combined_embedder_factory(self, text_proj_func):
        if not hasattr(self, 'cond_text_embedder'): return
        def caption_combined_embedder(c_mod):
            c_mod = text_proj_func(
                c_mod['prompt_embeds'],
                c_mod['prompt_attention_mask'],
                c_mod['prompt_embeds_2'],
                c_mod['prompt_attention_mask_2']
            )
            enc_mod = self.cond_text_embedder(c_mod)
            return enc_mod
        for k in ['local_caption', 'global_caption']:
            self.cond_text_embedder_combined = caption_combined_embedder

    def condition_encoding(self, data):
        enc_cond, mask_cond, pe_sinusoid, pe_rope = [], [], [], []
        for mod in self.modalities.keys():
            if mod in ['local_caption', 'global_caption']:
                mod_mask = data['mods_flags'][mod] == 0
                if (mod_mask == True).sum() == 0: continue  # do nothing when no prompts is available
                else:
                    # to hidden size
                    c_mod = data['conds'][mod]
                    enc_mod = self.cond_text_embedder_combined(c_mod)
                    enc_mod = self.mod_norm['caption'](enc_mod)
                    mod_ind = FeaturePositionalEncoding.mod_ind_table[mod]
                    zero_view_pe = True if mod == 'global_caption' else False   # global caption doesn't use view_pe, only mod_pe
                    enc_mod, pe_mod = self.pos_enc(
                        enc_mod, mod_ind=mod_ind, view_ids=data['view_id'], zero_view_pe=zero_view_pe, zero_patch_pe=True)  # (B, N*333, C)
                    _, _, seq_len, ndim = enc_mod.shape
                    mod_mask = mod_mask.unsqueeze(-1).repeat(1, 1, seq_len) 
                    if mod == 'global_caption':
                        # to fit the combination function, set to a multi-view version but the mask only use the first view
                        max_view = mod_mask.shape[1]
                        enc_mod = enc_mod.repeat(1, max_view, 1, 1) 
                        mod_mask[:, 1:] = False  
                        if pe_mod['sinusoid_all'] is not None: pe_mod['sinusoid_all'] = pe_mod['sinusoid_all'].repeat(1, max_view, 1, 1)
                        if pe_mod['rope'] is not None: pe_mod['rope'] = pe_mod['rope'].repeat(1, max_view, 1, 1)
            else:
                c_mod = data['conds'][mod]
                B, N, C, H, W = c_mod.shape
                mod_P = self.modalities[mod]['patch_size']
                mod_mask = data['mods_flags'][mod] == 0
                enc_mod_valid = self.embedders[mod](c_mod[mod_mask])#.reshape(B, N, self.hidden_size, H//mod_P, W//mod_P)
                enc_mod = torch.zeros(*mod_mask.shape, *enc_mod_valid.shape[1:], dtype=enc_mod_valid.dtype, device=enc_mod_valid.device)
                enc_mod[mod_mask] = enc_mod_valid
                enc_mod = enc_mod.permute(0, 1, 3, 4, 2)
                enc_mod = self.mod_norm[mod](enc_mod)
                mod_ind = FeaturePositionalEncoding.mod_ind_table[mod]
                enc_mod, pe_mod = self.pos_enc(enc_mod, mod_ind=mod_ind, view_ids=data['view_id'])  # (B, N*h*w, C)
                _, _, seq_len, ndim = enc_mod.shape
                mod_mask = mod_mask.unsqueeze(-1).repeat(1, 1, seq_len)

            enc_cond.append(enc_mod)
            mask_cond.append(mod_mask)
            pe_sinusoid.append(pe_mod['sinusoid_all'])
            pe_rope.append(pe_mod['rope'])
        
        enc_cond = torch.cat(enc_cond, dim=2)
        mask_cond = torch.cat(mask_cond, dim=2)
        pe_sinusoid = torch.cat(pe_sinusoid, dim=2) if pe_sinusoid[0] is not None else None
        pe_rope = torch.cat(pe_rope, dim=2) if pe_rope[0] is not None else None

        return enc_cond, mask_cond, pe_sinusoid, pe_rope
    

    def forward(self, t, data):
        """

        Args:
            x: Image/Ray features (B, N, C, H, W).
            t: Timesteps (B,).
            data: input data dict for each modality, including a data and a view_id mask

        Returns:
            (B, N, D, H, W)
        """

        enc_cond, mask_cond, pe_cond_sinusoid, pe_cond_rope = self.condition_encoding(data)  # (B, N*h*w, C)
        t = self.t_embedder(t)

        nt_enc_cond, seqlen_cond = full_to_padded(enc_cond, mask_cond)
        if pe_cond_sinusoid is not None:
            pe_cond_sinusoid, _ = full_to_padded(pe_cond_sinusoid, mask_cond)
        if pe_cond_rope is not None:
            pe_cond_rope, _ = full_to_padded(pe_cond_rope, mask_cond)

        # flash attention
        for i, block in enumerate(self.mv_encoder):
            block_args = (nt_enc_cond, t, pe_cond_sinusoid, pe_cond_rope, seqlen_cond)
            if self.training and self.grad_checkpoint:
                nt_enc_cond = checkpoint(block, *block_args, use_reentrant=False)
            else:
                nt_enc_cond = block(*block_args)

        return nt_enc_cond, seqlen_cond, pe_cond_sinusoid, pe_cond_rope


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        hidden_size=1152,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        modalities=None,
        qk_norm=None,
        encoder=False,
        encoder_depth=8,
        pe_config=None,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.depth = depth
        self.modalities = modalities
        self.training = True
        self.qk_norm = qk_norm
        self.encoder = encoder
        self.encoder_depth = encoder_depth
        self.pe_config = pe_config

        if encoder:
            self.encoder = Encoder(
                hidden_size=hidden_size,
                depth=encoder_depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                modalities=modalities,
                pe_config=pe_config,
                qk_norm=qk_norm,
            )

        # TODO move pretrained model name in config
        pretrained_model_name = 'Tencent-Hunyuan/HunyuanDiT-Diffusers'
        print(f'Initialize decoder with {pretrained_model_name}')
        self.decoder = HunyuanDiT2DModel.from_pretrained_with_holistic_modification(pretrained_model_name, subfolder='transformer')
        self.decoder.holistic_additional_modules(modalities, pe_config)

        self.encoder.caption_combined_embedder_factory(self.decoder.text_projection)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):  # TODO
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def forward(self, t, data):
        """

        Args:
            x: Image/Ray features (B, N, C, H, W).
            t: Timesteps (B,).
            data: input data dict for each modality, including a data and a view_id mask

        Returns:
            (B, N, D, H, W)
        """

        nt_enc_cond, seqlen_cond, pe_cond_sinusoid, pe_cond_rope = self.encoder(t, data)
        outputs = self.decoder(
            data, t,
            encoder_hidden_states=nt_enc_cond,
            pe_cond_sinusoid=pe_cond_sinusoid,
            pe_cond_rope=pe_cond_rope,
            seqlen_cond=seqlen_cond,
        )
   
        return outputs

