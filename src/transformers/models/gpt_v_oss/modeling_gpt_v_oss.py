# coding=utf-8
from typing import Optional, Union

import torch
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_gpt_v_oss import GptVOssConfig


class GptVOssProjector(nn.Module):
    def __init__(self, config: GptVOssConfig):
        super().__init__()
        in_dim = config.internvl_config.text_config.hidden_size
        hidden = config.projector_hidden_size
        out_dim = config.text_config.hidden_size
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class GptVOssPreTrainedModel(PreTrainedModel):
    config_class = GptVOssConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True


class GptVOssModel(GptVOssPreTrainedModel):
    def __init__(self, config: GptVOssConfig):
        super().__init__(config)
        self.vision_language_model = AutoModel.from_config(config.internvl_config)
        self.projector = GptVOssProjector(config)
        if getattr(config, 'model_merging', False):
            # hardy: for weight merging only
            self.oss_language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            self.oss_language_model = AutoModel.from_config(config.text_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.oss_language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.oss_language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.oss_language_model = decoder

    def get_decoder(self):
        return self.oss_language_model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, MoeModelOutputWithPast]:
        vision_outputs = self.vision_language_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = self.projector(vision_outputs.last_hidden_state)
        outputs = self.oss_language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        return outputs


class GptVOssForConditionalGeneration(GptVOssPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GptVOssConfig):
        super().__init__(config)
        self.model = GptVOssModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, MoeCausalLMOutputWithPast]:
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=getattr(outputs, "router_logits", None),
        )


__all__ = ["GptVOssModel", "GptVOssForConditionalGeneration", "GptVOssPreTrainedModel"]
