#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

# Copyright 2024 HunyuanDiT Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Optional, Union
import re
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import logging, SAFETENSORS_WEIGHTS_NAME, _get_model_file
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import (
    PatchEmbed,
    PixArtAlphaTextProjection,
    Timesteps,
    TimestepEmbedding,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.model_loading_utils import load_state_dict
from timm.models.vision_transformer import PatchEmbed as timmPatchEmbed

from .utils.nn import (
    HolisticAttnProcessor,
    FinalLayer,
    full_to_padded,
    padded_to_full,
    MultiLayerPatchEmbed,
    MultiLayerFinalLayer,
)
from .utils.pos_encoder import FeaturePositionalEncoding

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class HunyuanCombinedTimestepTextSizeStyleEmbedding(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim=1024, seq_len=256, cross_attention_dim=2048):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        # self.pooler = HunyuanDiTAttentionPool(
        #     seq_len, cross_attention_dim, num_heads=8, output_dim=pooled_projection_dim
        # )
        # # Here we use a default learned embedder layer for future extension.
        # self.style_embedder = nn.Embedding(1, embedding_dim)
        # extra_in_dim = 256 * 6 + embedding_dim + pooled_projection_dim
        # self.extra_embedder = PixArtAlphaTextProjection(
        #     in_features=extra_in_dim,
        #     hidden_size=embedding_dim * 4,
        #     out_features=embedding_dim,
        #     act_fn="silu_fp32",
        # )

    def forward(self, timestep, encoder_hidden_states, image_meta_size, style, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, 256)

        # # extra condition1: text
        # pooled_projections = self.pooler(encoder_hidden_states)  # (N, 1024)

        # # extra condition2: image meta size embdding
        # image_meta_size = get_timestep_embedding(image_meta_size.view(-1), 256, True, 0)
        # image_meta_size = image_meta_size.to(dtype=hidden_dtype)
        # image_meta_size = image_meta_size.view(-1, 6 * 256)  # (N, 1536)

        # # extra condition3: style embedding
        # style_embedding = self.style_embedder(style)  # (N, embedding_dim)

        # # Concatenate all extra vectors
        # extra_cond = torch.cat([pooled_projections, image_meta_size, style_embedding], dim=1)
        # conditioning = timesteps_emb + self.extra_embedder(extra_cond)  # [B, D]

        conditioning = timesteps_emb  # [B, D]

        return conditioning


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps
        ).to(origin_dtype)


class AdaLayerNormShift(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, elementwise_affine=True, eps=1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        shift = self.linear(self.silu(emb.to(torch.float32)).to(emb.dtype))
        x = self.norm(x) + shift.unsqueeze(dim=1)
        return x


@maybe_allow_in_graph
class HunyuanDiTBlock(nn.Module):
    r"""
    Transformer block used in Hunyuan-DiT model (https://github.com/Tencent/HunyuanDiT). Allow skip connection and
    QKNorm

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of headsto use for multi-head attention.
        cross_attention_dim (`int`,*optional*):
            The size of the encoder_hidden_states vector for cross attention.
        dropout(`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        activation_fn (`str`,*optional*, defaults to `"geglu"`):
            Activation function to be used in feed-forward. .
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            A small constant added to the denominator in normalization layers to prevent division by zero.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*):
            The size of the hidden layer in the feed-forward block. Defaults to `None`.
        ff_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the feed-forward block.
        skip (`bool`, *optional*, defaults to `False`):
            Whether to use skip connection. Defaults to `False` for down-blocks and mid-blocks.
        qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to use normalization in QK calculation. Defaults to `True`.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        cross_attention_dim: int = 1024,
        dropout=0.0,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-6,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        skip: bool = False,
        qk_norm: bool = True,
    ):
        super().__init__()

        # Define 3 blocks. Each block has its own normalization layer.
        # NOTE: when new version comes, check norm2 and norm 3
        # 1. Self-Attn
        self.norm1 = AdaLayerNormShift(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=True,
            processor=HolisticAttnProcessor(),
        )

        # 2. Cross-Attn
        self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=True,
            processor=HolisticAttnProcessor(),
        )
        # 3. Feed-forward
        self.norm3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,  ### 0.0
            activation_fn=activation_fn,  ### approx GeLU
            final_dropout=final_dropout,  ### 0.0
            inner_dim=ff_inner_dim,  ### int(dim * mlp_ratio)
            bias=ff_bias,
        )

        # 4. Skip Connection
        if skip:
            self.skip_norm = FP32LayerNorm(2 * dim, norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_sinusoid_emb=None,
        enc_sinusoid_emb=None,
        image_rotary_emb=None,
        enc_rotary_emb=None,
        image_seqlen=None,
        enc_seqlen=None,
        skip=None,
        uncond=False,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat([hidden_states, skip], dim=-1)
            cat = self.skip_norm(cat)
            hidden_states = self.skip_linear(cat)

        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states, temb)  ### checked: self.norm1 is correct
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=norm_hidden_states,

            query_pos_s=image_sinusoid_emb,
            key_pos_s=image_sinusoid_emb,
            query_pos_r=image_rotary_emb,
            key_pos_r=image_rotary_emb,
            seqlen_q=image_seqlen,
            seqlen_kv=image_seqlen,
        )
        hidden_states = hidden_states + attn_output

        # 2. Cross-Attention
        if isinstance(uncond, bool):
            uncond = torch.full((hidden_states.shape[0],), uncond, dtype=torch.bool, device=hidden_states.device)
        hidden_states = hidden_states + (~uncond).to(hidden_states.dtype)[:, None, None] * self.attn2(
            self.norm2(hidden_states),
            encoder_hidden_states=encoder_hidden_states,

            query_pos_s=image_sinusoid_emb,
            key_pos_s=enc_sinusoid_emb,
            query_pos_r=image_rotary_emb,
            key_pos_r=enc_rotary_emb,
            seqlen_q=image_seqlen,
            seqlen_kv=enc_seqlen,
        )

        # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
        mlp_inputs = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(mlp_inputs)

        return hidden_states


class HunyuanDiT2DModel(ModelMixin, ConfigMixin):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88):
            The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        patch_size (`int`, *optional*):
            The size of the patch to use for the input.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward.
        sample_size (`int`, *optional*):
            The width of the latent images. This is fixed during training since it is used to learn a number of
            position embeddings.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        cross_attention_dim (`int`, *optional*):
            The number of dimension in the clip text embedding.
        hidden_size (`int`, *optional*):
            The size of hidden layer in the conditioning embedding layers.
        num_layers (`int`, *optional*, defaults to 1):
            The number of layers of Transformer blocks to use.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the hidden layer size to the input size.
        learn_sigma (`bool`, *optional*, defaults to `True`):
             Whether to predict variance.
        cross_attention_dim_t5 (`int`, *optional*):
            The number dimensions in t5 text embedding.
        pooled_projection_dim (`int`, *optional*):
            The size of the pooled projection.
        text_len (`int`, *optional*):
            The length of the clip text embedding.
        text_len_t5 (`int`, *optional*):
            The length of the T5 text embedding.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "gelu-approximate",
        sample_size=32,
        hidden_size=1152,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        cross_attention_dim: int = 1024,
        norm_type: str = "layer_norm",
        cross_attention_dim_t5: int = 2048,
        pooled_projection_dim: int = 1024,
        text_len: int = 77,
        text_len_t5: int = 256,
    ):
        super().__init__()
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim

        self.text_embedder = PixArtAlphaTextProjection(
            in_features=cross_attention_dim_t5,
            hidden_size=cross_attention_dim_t5 * 4,
            out_features=1024,  # TODO channel is hard coded
            act_fn="silu_fp32",
        )

        self.text_embedding_padding = nn.Parameter(
            torch.randn(text_len + text_len_t5, 1024, dtype=torch.float32)  # TODO channel is hard coded
        )

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            patch_size=patch_size,
            pos_embed_type=None,
        )

        self.time_extra_emb = HunyuanCombinedTimestepTextSizeStyleEmbedding(
            hidden_size,
            pooled_projection_dim=pooled_projection_dim,
            seq_len=text_len_t5,
            cross_attention_dim=cross_attention_dim_t5,
        )

        # HunyuanDiT Blocks
        self.blocks = nn.ModuleList(
            [
                HunyuanDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    activation_fn=activation_fn,
                    ff_inner_dim=int(self.inner_dim * mlp_ratio),
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                    skip=layer > num_layers // 2,
                )
                for layer in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(HolisticAttnProcessor())

    def holistic_additional_modules(self, modalities, pe_config):
        self.grad_checkpoint = False
        self.modalities = modalities
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        # grid size of all 2D modalities should be identical
        grid_sizes = [v['width'] // v['patch_size'] for k, v in modalities.items() if v['dimensions'] == 2]
        assert all([grid_sizes[0] == it for it in grid_sizes])
        grid_size = grid_sizes[0]
        # Init PatchEmbedders
        # PE (view_id, token pos, modality type)
        self.pos_enc = FeaturePositionalEncoding(pe_config, hidden_size, num_heads, grid_size)
        # modality-specific embedders
        self.mod_embedders = {}
        for mod, mod_conf in modalities.items():
            if mod_conf['dimensions'] != 2 or mod == 'rgb': continue
            if mod == 'depth' and mod_conf['patch_size'] > 4:
                self.mod_embedders[mod] = MultiLayerPatchEmbed(
                    img_size=mod_conf['width'],
                    patch_size=mod_conf['patch_size'],
                    in_chans=sum(mod_conf['gen_channel'].values()) + sum(mod_conf.get('gen_aux_channel', {}).values()),
                    embed_dim=hidden_size,
                )
            else:
                self.mod_embedders[mod] = timmPatchEmbed(
                    img_size=mod_conf['width'],
                    patch_size=mod_conf['patch_size'],
                    in_chans=sum(mod_conf['gen_channel'].values()) + sum(mod_conf.get('gen_aux_channel', {}).values()),
                    embed_dim=hidden_size,
                    bias=True,
                    flatten=False,
                )
        self.mod_embedders = nn.ModuleDict(self.mod_embedders)
        # modality-specific normalization layer
        self.mod_norm = nn.ModuleDict({mod: nn.LayerNorm(hidden_size) for mod, mod_conf in modalities.items() if mod_conf['dimensions'] == 2})
        # modality-specific final layers
        self.mod_final_layers = {}
        for mod, mod_conf in modalities.items():
            if mod_conf['dimensions'] != 2 or mod == 'rgb': continue
            if mod == 'depth' and mod_conf['patch_size'] > 4:
                self.mod_final_layers[mod] = MultiLayerFinalLayer(
                    hidden_size,
                    mod_conf['patch_size'],
                    sum(mod_conf['gen_channel'].values()),
                )
            else:
                self.mod_final_layers[mod] = FinalLayer(
                    hidden_size,
                    mod_conf['patch_size'],
                    sum(mod_conf['gen_channel'].values()),
                )
        self.mod_final_layers = nn.ModuleDict(self.mod_final_layers)
        # init
        for mod in modalities.keys():
            if modalities[mod]['dimensions'] != 2 or mod == 'rgb': continue

            embedder = self.mod_embedders[mod]
            if isinstance(embedder, MultiLayerPatchEmbed):
                embedder.initialize_weights()
            else:
                w = embedder.proj.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
                nn.init.constant_(embedder.proj.bias, 0)

            self.mod_final_layers[mod].initialize_weights()

    @classmethod
    def from_pretrained_with_holistic_modification(cls, url, subfolder='transformer'):
        config, unused_kwargs, commit_hash = cls.load_config(
            url,
            return_unused_kwargs=True,
            return_commit_hash=True,
            subfolder=subfolder,
        )
        # edit cross attn channel num
        # config['cross_attention_dim'] = config['hidden_size']
        # TODO remove var output
        model_file = _get_model_file(
            url,
            weights_name=SAFETENSORS_WEIGHTS_NAME,
            subfolder=subfolder,
            commit_hash=commit_hash,
        )
        model = cls.from_config(config, **unused_kwargs)
        state_dict = load_state_dict(model_file)
        model._convert_deprecated_attention_blocks(state_dict)
        # remove cross attn weights
        # cross_attn_pattern = re.compile(r'blocks..*.attn2.*')
        # state_dict = {k: v for k, v in state_dict.items() if not re.match(cross_attn_pattern, k)}
        # TODO remove var output
        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
            model,
            state_dict,
            model_file,
            url,
            ignore_mismatched_sizes=True,
        )
        # zero-out cross attn
        for b in model.blocks:
            nn.init.zeros_(b.attn2.to_out[0].weight)
            nn.init.zeros_(b.attn2.to_out[0].bias)
        return model

    def apply_embedder(self, data):
        enc, mask, pe_sinusoid, pe_rope = [], [], [], []
        mask_dict = {}
        for mod in self.modalities.keys():
            if mod in ['local_caption', 'global_caption']: continue

            c_mod = data['gens'][mod]
            B, N, C, H, W = c_mod.shape
            mod_P = self.modalities[mod]['patch_size']
            mod_mask = data['mods_flags'][mod] == 1
            if mod == 'rgb':
                enc_mod_valid = self.pos_embed(c_mod[mod_mask]).permute(0,2,1).unflatten(2, (H//mod_P, W//mod_P))
            else:
                enc_mod_valid = self.mod_embedders[mod](c_mod[mod_mask])#.reshape(B, N, self.config.hidden_size, H//mod_P, W//mod_P)
            enc_mod = torch.zeros(*mod_mask.shape, *enc_mod_valid.shape[1:], dtype=enc_mod_valid.dtype, device=enc_mod_valid.device)
            enc_mod[mod_mask] = enc_mod_valid
            enc_mod = enc_mod.permute(0, 1, 3, 4, 2)
            enc_mod = self.mod_norm[mod](enc_mod)
            mod_ind = FeaturePositionalEncoding.mod_ind_table[mod]
            enc_mod, pe_mod = self.pos_enc(enc_mod, mod_ind=mod_ind, view_ids=data['view_id'])  # (B, N*h*w, C)
            _, _, seq_len, ndim = enc_mod.shape
            mod_mask = mod_mask.unsqueeze(-1).repeat(1, 1, seq_len)

            enc.append(enc_mod)
            mask.append(mod_mask)
            pe_sinusoid.append(pe_mod['sinusoid_all'])
            pe_rope.append(pe_mod['rope'])
            mask_dict[mod] = mod_mask
        
        enc = torch.cat(enc, dim=2)
        mask = torch.cat(mask, dim=2)
        pe_sinusoid = torch.cat(pe_sinusoid, dim=2) if pe_sinusoid[0] is not None else None
        pe_rope = torch.cat(pe_rope, dim=2) if pe_rope[0] is not None else None

        return enc, mask, pe_sinusoid, pe_rope, mask_dict, B, N

    def unpatchify(self, x, p=None, c=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        if p is not None and c is None:
            c = x.shape[-1] // p // p
        elif p is None and c is not None:
            p = int((x.shape[-1] / c)**0.5)
        elif p is None and c is None:
            raise ValueError('patch size and channel cannot be both None')
        assert x.shape[-1] == p*p*c

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.grad_checkpoint = True

    def func_with_grad_checkpointing(self, func, *args, **kwargs):
        if self.grad_checkpoint:
            return checkpoint(func, *args, use_reentrant=False, **kwargs)
        else:
            return func(*args, **kwargs)

    def text_projection(self, encoder_hidden_states, text_embedding_mask, encoder_hidden_states_t5, text_embedding_mask_t5):
        # text projection
        batch_size, num_view, sequence_length, _ = encoder_hidden_states_t5.shape
        encoder_hidden_states_t5 = self.text_embedder(
            encoder_hidden_states_t5.view(-1, encoder_hidden_states_t5.shape[-1])
        )
        encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, num_view, sequence_length, -1)
        
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_t5], dim=2)
        text_embedding_mask = torch.cat([text_embedding_mask, text_embedding_mask_t5], dim=-1)
        text_embedding_mask = text_embedding_mask.unsqueeze(-1).bool()
        encoder_hidden_states = torch.where(text_embedding_mask, encoder_hidden_states, self.text_embedding_padding)
        return encoder_hidden_states

    def forward(
        self,
        data,
        timestep,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        pe_cond_sinusoid=None,
        pe_cond_rope=None,
        seqlen_cond=None,
        return_dict=True,
    ):
        """
        The [`HunyuanDiT2DModel`] forward method.

        Args:
        hidden_states (`torch.Tensor` of shape `(batch size, dim, height, width)`):
            The input tensor.
        timestep ( `torch.LongTensor`, *optional*):
            Used to indicate denoising step.
        encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer. This is the output of `BertModel`.
        text_embedding_mask: torch.Tensor
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
            of `BertModel`.
        encoder_hidden_states_t5 ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer. This is the output of T5 Text Encoder.
        text_embedding_mask_t5: torch.Tensor
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
            of T5 Text Encoder.
        image_meta_size (torch.Tensor):
            Conditional embedding indicate the image sizes
        style: torch.Tensor:
            Conditional embedding indicate the style
        image_rotary_emb (`torch.Tensor`):
            The image rotary embeddings to apply on query and key tensors during attention calculation.
        return_dict: bool
            Whether to return a dictionary.
        """

        enc_gen, mask_gen, pe_gen_sinusoid, pe_gen_rope, mask_gen_dict, B, N = self.apply_embedder(data)

        temb = self.time_extra_emb(timestep, None, None, None, hidden_dtype=encoder_hidden_states.dtype)  # [B, D]

        hidden_states, seqlen_gen = full_to_padded(enc_gen, mask_gen)
        if pe_gen_sinusoid is not None:
            pe_gen_sinusoid, _ = full_to_padded(pe_gen_sinusoid, mask_gen)
        if pe_gen_rope is not None:
            pe_gen_rope, _ = full_to_padded(pe_gen_rope, mask_gen)

        # temb = self.time_extra_emb(
        #     timestep, encoder_hidden_states_t5, image_meta_size, style, hidden_dtype=timestep.dtype
        # )  # [B, D]

        # text projection
        # encoder_hidden_states = self.text_projection(encoder_hidden_states, text_embedding_mask, encoder_hidden_states_t5, text_embedding_mask_t5)

        skips = []
        for layer, block in enumerate(self.blocks):
            skip_pop = (layer > self.config.num_layers // 2)
            if layer > self.config.num_layers // 2:
                pass
            hidden_states = self.func_with_grad_checkpointing(
                block,
                hidden_states,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                image_sinusoid_emb=pe_gen_sinusoid,
                # enc_sinusoid_emb=pe_cond_sinusoid,
                image_rotary_emb=pe_gen_rope,
                # enc_rotary_emb=pe_cond_rope,
                image_seqlen=seqlen_gen,
                enc_seqlen=seqlen_cond,
                skip=skips.pop() if layer > self.config.num_layers // 2 else None,
                uncond=data.get('uncond', False),
            )  # (N, L, D)
            if layer < (self.config.num_layers // 2 - 1):
                skips.append(hidden_states)

        x, _ = padded_to_full(hidden_states, seqlen_gen, mask_gen)
        x = x.reshape(B, N, -1, self.config.hidden_size)  # (B, N, seq_len, D)

        # final layer
        outputs = {}
        for mod in self.modalities:
            if mod in ['local_caption', 'global_caption']: continue

            mod_map_size = self.modalities[mod]['width']//self.modalities[mod]['patch_size']
            mod_token_len = mod_map_size**2
            x_mod, x = torch.split(x, (mod_token_len, x.shape[2]-mod_token_len), dim=2)
            if mod == 'rgb':
                mod_out = self.norm_out(x_mod.flatten(1,2), temb.to(torch.float32)).unflatten(1, (N, -1)).flatten(0, 1)
                mod_out = self.proj_out(mod_out)
                mod_out = self.unpatchify(mod_out, p=self.modalities[mod]['patch_size']).unflatten(0, (B, N))  # (B, N, C, H, W)
                mod_out, _ = mod_out.unflatten(2, (2, -1)).unbind(2)  # discard variance
            elif mod == 'depth' and isinstance(self.mod_final_layers[mod], MultiLayerFinalLayer):
                mod_out = x_mod.unflatten(2, (mod_map_size, mod_map_size))  # (B, N, H, W, D)
                mod_out = self.mod_final_layers[mod](mod_out, temb)  # (B, N, D, H, W)
            else:
                mod_out = self.mod_final_layers[mod](x_mod.flatten(1, 2), temb).unflatten(1, (N, -1)).flatten(0, 1)  # (B, T, D) -> (B*N, H*W, D)
                mod_out = self.unpatchify(mod_out, p=self.modalities[mod]['patch_size']).unflatten(0, (B, N))  # (B, N, C, H, W)

            outputs.update({mod: mod_out, f'{mod}_mask': mask_gen_dict[mod][..., 0]})

        return outputs


    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)
