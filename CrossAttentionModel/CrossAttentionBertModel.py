import torch
from transformers import BertModel
from torch.nn import Linear, Dropout, Sigmoid, Module


class CrossAttentionBertClassificationHead(Module):
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


class CrossAttentionBertModel(BertModel):
    def __init__(self, config):
        super(CrossAttentionBertModel, self).__init__(config)
        self.classifier = CrossAttentionBertClassificationHead(config)
        self.config = config
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
    ):
        batch_size, seq_length = input_ids.shape

        cross_attention_mask = torch.zeros(
            (batch_size, seq_length, seq_length),
            dtype=torch.long,
            device=input_ids.device,
        )

        for i in range(batch_size):
            cross_attention_mask[i, :, :] = attention_mask[i].view(
                -1, 1
            ) * attention_mask[i].view(1, -1)

        outputs = super(CrossAttentionBertModel, self).forward(
            input_ids,
            cross_attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
        )

        return (
            None,
            self.classifier(outputs.last_hidden_state[:, 0, :]).squeeze(),
        )
