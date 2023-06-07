from transformers import Trainer


class ExtractiveSummarizatorTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss
