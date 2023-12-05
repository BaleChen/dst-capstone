from dataclasses import dataclass, field
import logging
import pathlib
import typing
import os

# Bale Chen
import pdb
from time import gmtime, strftime

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import Trainer, BitsAndBytesConfig, deepspeed, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from torch.nn import CrossEntropyLoss

from fastchat.train.train import (
    ModelArguments,
)

from dataset_utils import (
    DataArguments,
    make_supervised_data_module,
) # Bale Chen

from utils import (
    prepare_exp_name,
)

from fastchat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

class LlamaWithUnlikelihoodLoss(LlamaForCausalLM):
    def __init__(self, config, unlikely_coef=0.0):
        super().__init__(config)
        self.unlikely_coef = unlikely_coef
    
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        unlikely_input_ids = None,
        unlikely_attention_mask = None,
        unlikely_targets = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if unlikely_input_ids is not None:
            # unlikely_input_ids (Batch_size, Beam_size, Seq_len)
            for i in range(unlikely_input_ids.shape[1]):
                unlikely_input_ids_i = unlikely_input_ids[:, i, :].to(logits.device)
                unlikely_attention_mask_i = unlikely_attention_mask[:, i, :].to(logits.device)
                unlikely_target_i = unlikely_targets[:, i, :] 
                unlikely_output = self.model(
                    input_ids=unlikely_input_ids_i,
                    attention_mask=unlikely_attention_mask_i,
                )
                unlikely_hidden_states = unlikely_output[0]
                unlikely_logits = self.lm_head(unlikely_hidden_states).float()
                unlikely_loss_fct = CrossEntropyLoss()

                shift_unlikely_logits = unlikely_logits[..., :-1, :].contiguous()
                shift_unlikely_target = unlikely_target[..., 1:].contiguous()
                shift_unlikely_logits = shift_unlikely_logits.view(-1, self.config.vocab_size)
                shift_unlikely_target = shift_unlikely_target.view(-1)
                unlikely_target = unlikely_target.to(unlikely_logits.device)
                unlikely_loss = -1 * self.unlikely_coef * unlikely_loss_fct(shift_unlikely_logits, shift_unlikely_target)
                loss += unlikely_loss / unlikely_input_ids.shape[1]

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )