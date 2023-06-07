import torch
from torch.nn import Linear, Dropout, Tanh, Module
from transformers import AutoModel, BigBirdPreTrainedModel
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


class ExtractiveSummarizatorClassificationHead(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = Dropout(self.config.classifier_dropout)
        self.out_proj = Linear(self.config.hidden_size, self.config.num_labels)
        """
        The tanh activation function is applied before the final linear layer in the ClassificationHead to introduce non-linearity into the model. 
        Without an activation function, the ClassificationHead would consist of a series of linear transformations,
        which could limit its ability to learn complex relationships between the input features and the target classes.
        The tanh activation function is a non-linear function that squashes its input into the range (-1, 1).
        It can help the model to learn more complex decision boundaries by allowing it to combine the input features in non-linear ways.
        Linearity refers to a relationship between two variables where a change in one variable results in a proportional change in the other variable. 
        For example, if you double the value of one variable, the value of the other variable also doubles. Linear relationships can be represented by a straight line on a graph.
        In contrast, non-linearity refers to a relationship between two variables where a change in one variable does not result in a proportional change in the other variable.
        Non-linear relationships can take many different forms and can be represented by curved lines or more complex shapes on a graph.
        """
        self.actv_fn = Tanh()

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.actv_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ExtractiveSummarizatorModel(BigBirdPreTrainedModel):
    def __init__(self, config, bce_pos_weight=None):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config)
        self.mhattention = torch.nn.MultiheadAttention(
            self.encoder.config.hidden_size,
            self.encoder.config.num_attention_heads,
        )
        self.classifier = ExtractiveSummarizatorClassificationHead(config)
        self.config = config
        self.bce_pos_weight = bce_pos_weight
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
        labels=None,
    ):
        encoder_outputs = self.encoder(
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

        """
        The hidden_states tensor is transposed to match the expected input shape of the MultiheadAttention layer. 
        The MultiheadAttention layer expects its inputs to have shape (sequence_length, batch_size, hidden_size), 
        while the hidden_states tensor returned by the encoder has shape (batch_size, sequence_length, hidden_size). 
        Transposing the first two dimensions of the hidden_states tensor changes its shape to (sequence_length, batch_size, hidden_size),
        which matches the expected input shape of the MultiheadAttention layer.
        """
        hidden_states = encoder_outputs.last_hidden_state.transpose(0, 1)

        """
        The MultiheadAttention layer takes three inputs: query, key, and value.
        It computes attention weights between the query and key tensors and uses these weights to compute a weighted average of the value tensor.
        In our case, we want to compute attention weights between all pairs of sentences in the batch, so we pass the same tensor (hidden_states) as both the query and key inputs.
        We also want to compute a weighted average of the hidden states of all sentences in the batch, so we pass the same tensor (hidden_states) as the value input.
        
        """
        attended_states, attention_weights = self.mhattention(
            hidden_states, hidden_states, hidden_states
        )

        # Get attended state of [CLS] token
        cls_attended_state = attended_states[0]

        # Pass pooled_output through classifier to get logits
        logits = self.classifier(
            cls_attended_state,
        )

        loss = None

        if labels is not None:
            logits = logits.squeeze()
            if labels.numel() == 1:
                labels = labels.squeeze()
            if self.bce_pos_weight is not None:
                loss_fn = torch.nn.BCEWithLogitsLoss(weight=self.bce_pos_weight)
            else:
                loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
            encoder_outputs.logits = logits

        return (
            SequenceClassifierOutputWithPoolingAndCrossAttentions(
                loss=loss,
                last_hidden_state=encoder_outputs.last_hidden_state,
                pooler_output=encoder_outputs.pooler_output,
                hidden_states=encoder_outputs.hidden_states,
                past_key_values=encoder_outputs.past_key_values,
                attentions=encoder_outputs.attentions,
                logits=encoder_outputs.logits,
                cross_attentions=attention_weights,
            )
            if loss is not None
            else encoder_outputs
        )
