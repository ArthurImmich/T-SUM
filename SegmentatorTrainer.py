from transformers import Trainer
import torch


class SegmentatorTrainer(Trainer):
    def __init__(self, device, *args, **kwargs):
        self.device = device
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").squeeze().float()
        loss_fct = torch.nn.BCELoss()
        outputs = model(**inputs)
        loss = loss_fct(outputs[1], labels)
        return (loss, outputs) if return_outputs else loss
