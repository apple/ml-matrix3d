#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from typing import Optional, Tuple, Union, Dict, List
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
from transformers.models.dinov2.modeling_dinov2 import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    Dinov2SelfAttention,
    Dinov2Layer,
    Dinov2PreTrainedModel,
    Dinov2Embeddings,
    Dinov2PatchEmbeddings,
)

from .dinov2_adaln.adaln import AdaLayerNorm

class Dinov2SelfAttentionSDP(Dinov2SelfAttention):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)
        self.dropout_prob = config.attention_probs_dropout_prob
        assert self.dropout_prob == 0.0

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        assert head_mask is None
        assert not output_attentions

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, dropout_p=self.dropout_prob)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)

        return outputs

class AdaDinov2Layer(Dinov2Layer):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)
        self.norm1 = AdaLayerNorm(config.hidden_size, eps=config.layer_norm_eps, mod_act=config.hidden_act)
        self.norm2 = AdaLayerNorm(config.hidden_size, eps=config.layer_norm_eps, mod_act=config.hidden_act)
        self.attention.attention = Dinov2SelfAttentionSDP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        modulation: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.norm1(hidden_states, modulation),  # in Dinov2, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states, modulation)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = layer_output + hidden_states

        outputs = (layer_output,) + outputs

        return outputs

class AdaDinov2Encoder(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([AdaDinov2Layer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        modulation: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_inputs = (hidden_states, layer_head_mask, modulation, output_attentions)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    *layer_inputs
                )
            else:
                layer_outputs = layer_module(*layer_inputs)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class AdaDinov2PreTrainedModel(Dinov2PreTrainedModel):
    def _init_weights(self, module: nn.Linear | nn.Conv2d | nn.LayerNorm | AdaLayerNorm) -> None:
        super()._init_weights(module)
        if isinstance(module, AdaLayerNorm):
            module.mod_init()

class AdaDinov2Model(AdaDinov2PreTrainedModel):
    def __init__(self, config: Dinov2Config):
        super().__init__(config)
        self.config = config

        self.embeddings = Dinov2Embeddings(config)
        self.encoder = AdaDinov2Encoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # TODO whether change this to ada

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> Dinov2PatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def set_patch_size(self, new_size):
        # NOTE call immediately after from_pretrained
        # NOTE error is large (~0.5), not used for now, input size is 224 or 448, patch size remain 14
        num_patch = self.config.image_size // self.config.patch_size
        self.config.patch_size = new_size
        self.config.image_size = new_size * num_patch
        new_projection = nn.Conv2d(
            self.config.num_channels,
            self.config.hidden_size,
            kernel_size=new_size,
            stride=new_size).eval()
        with torch.no_grad():
            new_projection.bias[:] = self.embeddings.patch_embeddings.projection.bias
            new_projection.weight[:] = F.interpolate(
                self.embeddings.patch_embeddings.projection.weight,
                new_size, mode='bilinear', align_corners=False,
            )
        self.embeddings.patch_embeddings.projection = new_projection

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        modulation: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            modulation=modulation,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
