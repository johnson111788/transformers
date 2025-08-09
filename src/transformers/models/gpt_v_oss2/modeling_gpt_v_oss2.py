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
from .configuration_gpt_v_oss2 import GptVOss2Config

import pdb

class GptVOss2Projector(nn.Module):
    def __init__(self, config: GptVOss2Config):
        super().__init__()
        in_dim = config.internvl_config.text_config.hidden_size
        hidden = config.projector_hidden_size
        out_dim = config.text_config.hidden_size
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class GptVOss2PreTrainedModel(PreTrainedModel):
    config_class = GptVOss2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

from torch.nn.utils.rnn import pad_sequence



class GptVOss2Model(GptVOss2PreTrainedModel):
    def __init__(self, config: GptVOss2Config):
        super().__init__(config)
        self.vision_language_model = AutoModel.from_config(config.internvl_config)
        self.projector = GptVOss2Projector(config)
        self.oss_language_model = AutoModel.from_config(config.text_config)

        if getattr(config, 'model_merging', False):
            # hardy: for weight merging only
            print(f'Performing model merging!!!!! This shouldn\'t happen at training!')
            pdb.set_trace()
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
    
    def get_vision_language_tokenizer(self):
        return self.vision_language_tokenizer
    
    def get_language_tokenizer(self):
        return self.language_tokenizer
    
    def get_tokenizer(self):
        return self.tokenizer


    def masked_select_and_pad(self, input_ids: torch.Tensor,
                            mask: torch.Tensor,
                            padding_value: int = 0):
        """
        Args
        ----
        input_ids : LongTensor of shape (bsz, seqlen)
        mask      : same shape, 0/1 or bool
        padding_value : what to right-pad with

        Returns
        -------
        padded_ids      : LongTensor (bsz, max_len)
        new_attn_mask   : LongTensor (bsz, max_len) — 1 for real tokens, 0 for pad
        """
        # keep only the tokens the mask says are “on”
        kept = [row_ids[row_mask.bool()]          # 1-D tensor, variable length
                for row_ids, row_mask in zip(input_ids, mask)]


        # right-pad to the longest sequence in the batch
        padded_ids = pad_sequence(
            kept, batch_first=True, padding_value=padding_value
        )

        # attention mask: 1 where not pad, 0 where pad
        new_attn_mask = (padded_ids != padding_value).long()

        return padded_ids, new_attn_mask
    
    def unpad_tensor(self, padded: torch.Tensor,
                    attn_mask: torch.Tensor):
        """
        Args
        ----
        padded      : Tensor (bsz, seqlen, dim)  – the right-padded batch
        attn_mask   : Tensor (bsz, seqlen)       – 1 for real tokens, 0 for pad

        Returns
        -------
        list_of_tensors : length == bsz
            Each element i has shape (valid_len_i, dim), where
            valid_len_i == attn_mask[i].sum().
        """
        # Ensure boolean mask
        attn_mask_bool = attn_mask.bool()

        # Split row-by-row, selecting un-padded positions
        unpadded = [
            row[mask_row]           # keeps only the “real” time-steps
            for row, mask_row in zip(padded, attn_mask_bool)
        ]

        return unpadded


    def reverse_and_gather_vision_tokens(self, input_ids: torch.LongTensor):

        vision_keys = ['start_image_token', 'end_image_token', 'context_image_token', 'video_token']
        processed_ids = input_ids.clone()
        for vk in vision_keys:
            vk_id = self.get_tokenizer().convert_tokens_to_ids(getattr(self.get_tokenizer(), vk))
            # old id
            vk_vl_id = self.get_vision_language_tokenizer().convert_tokens_to_ids(getattr(self.get_vision_language_tokenizer(), vk)) 
            # get mask
            _mask = input_ids == vk_id
            # replace with old id
            processed_ids = processed_ids.masked_fill(_mask, vk_vl_id) 

        vision_mask = input_ids!=processed_ids
        processed_ids, new_mask = self.masked_select_and_pad(processed_ids, vision_mask, self.get_vision_language_tokenizer().pad_token_id)

        return processed_ids, new_mask, vision_mask
    

    def scatter_sources_into_target(self, sources: list[torch.Tensor],          # list of 1-D tensors / lists
                                    target: torch.Tensor,
                                    mask:   torch.Tensor):
        """
        Fills `target` where `mask == 1` with the corresponding values from `sources`.

        Args
        ----
        sources : list[Tensor] / list[list[int]]  (len == bsz)
                len(sources[i]) must equal (mask[i] == 1).sum()
        target  : Tensor  (bsz, seqlen)
        mask    : Tensor  (bsz, seqlen) – 1 → position to replace, 0 → keep

        Returns
        -------
        Tensor (bsz, seqlen) – same shape as `target`, but with scattered values.
        """
        out = target.clone()

        for row_idx, (row_src, row_mask) in enumerate(zip(sources, mask)):
            insert_pos = row_mask.bool()              # 1 → True, 0 → False
            if insert_pos.sum().item() != len(row_src):
                raise ValueError(f"Row {row_idx}: mask wants "
                                f"{insert_pos.sum().item()} values but source has "
                                f"{len(row_src)}")
            out[row_idx, insert_pos] = torch.as_tensor(
                row_src, dtype=target.dtype, device=target.device
            )

        return out
    # def reverse_vision_ids(self, input_ids: torch.LongTensor):
    #     vision_keys = ['start_image_token', 'end_image_token', 'context_image_token', 'video_token']
    #     new_ids = input_ids.clone()
    #     for vk in vision_keys:
    #         vk_id = self.get_tokenizer().convert_tokens_to_ids(getattr(self.get_tokenizer(), vk))
    #         # old id
    #         vk_vl_id = self.get_tokenizer().convert_tokens_to_ids(getattr(self.get_vision_language_tokenizer(), vk)) 
    #         # get mask
    #         _mask = new_ids == vk_id
    #         # replace with the old id
    #         new_ids = new_ids.masked_fill(_mask, vk_vl_id) 

    #     return new_ids

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

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # get oss emb
        if inputs_embeds is None:
            input_embeds = self.get_input_embeddings()(input_ids)


        # pdb.set_trace()
        inputs_embeds; pixel_values

        if pixel_values is not None:
            # replace image tokens in input_ids with lang_token
            input_ids_for_vl_model, attention_mask_for_vl_model, vision_mask = self.reverse_and_gather_vision_tokens(input_ids)
        
            # get image features
            vision_outputs = self.vision_language_model(
                input_ids=input_ids_for_vl_model,
                pixel_values=pixel_values,
                attention_mask=attention_mask_for_vl_model, # all ones
                # **kwargs,
            )

            # this is a padded tensor!
            vision_feature = self.projector(vision_outputs.last_hidden_state)
            # unpadded list[tensor]
            vision_feature_unpadded_lst = self.unpad_tensor(vision_feature, attention_mask_for_vl_model)
        

            # scatter image feature to input_embeddings
            input_embeds = self.scatter_sources_into_target(
                sources=vision_feature_unpadded_lst,
                target=input_embeds,
                mask=vision_mask
            )
        
        # forward language model
        # cache handled inside
        outputs = self.oss_language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids, 
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        return outputs


class GptVOss2ForConditionalGeneration(GptVOss2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GptVOss2Config):
        super().__init__(config)
        self.model = GptVOss2Model(config)
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
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, MoeCausalLMOutputWithPast]:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
        )
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
            output_router_logits=output_router_logits,
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



    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # pdb.set_trace()
        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

__all__ = ["GptVOss2Model", "GptVOss2ForConditionalGeneration", "GptVOss2PreTrainedModel"]
