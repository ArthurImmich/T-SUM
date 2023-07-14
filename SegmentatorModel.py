import torch
from torch.nn import Linear, Dropout, Sigmoid, Module
from transformers import BertModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SequenceClassifierOutputWithPoolingAndCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class SegmentatorClassificationHead(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = Dropout(self.config.classifier_dropout)
        self.out_proj = Linear(self.config.hidden_size, self.config.num_labels)
        self.actv_fn = Sigmoid()

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = self.actv_fn(x)
        return x


class SegmentatorModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.mhattention = torch.nn.MultiheadAttention(
            self.encoder.config.hidden_size,
            self.encoder.config.num_attention_heads,
            batch_first=True,
            dropout=config.classifier_dropout,
        )
        self.classifier = SegmentatorClassificationHead(config)
        self.config = config
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=True,
        output_hidden_states=True,
    ):
        encoder_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        cls_last_hidden_state = encoder_outputs.last_hidden_state[:, 0, :]

        attended_states, _ = self.mhattention(
            cls_last_hidden_state,
            cls_last_hidden_state,
            cls_last_hidden_state,
        )

        logits = self.classifier(attended_states).squeeze()

        return (None, logits.squeeze())
