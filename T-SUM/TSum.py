from transformers import (
    AutoModelForSequenceClassification,
    BigBirdPegasusForConditionalGeneration,
    AutoTokenizer,
)
from transformers.models.bigbird_pegasus import BigBirdPegasusForConditionalGeneration
from transformers.tokenization_utils import PaddingStrategy
from datasets import Dataset
from segmentator.SegmentatorModel import SegmentatorModel
from extractor.ExtractorModel import ExtractorModel
import torch


class TSum:
    def __init__(self, device, context_window):
        self.context_window = context_window
        self.device = device

        self.segmentatorModel = SegmentatorModel.from_pretrained("./segmentator/model")
        self.extractorModel = ExtractorModel.from_pretrained("./extractor/model")
        self.abstractorModel = BigBirdPegasusForConditionalGeneration.from_pretrained(
            "./abstractor/model"
        )

        self.segmentatorTokenizer = AutoTokenizer.from_pretrained("./segmentator/model")
        self.extractorTokenizer = AutoTokenizer.from_pretrained("./extractor/model")
        self.abstractorTokenizer = AutoTokenizer.from_pretrained("./abstractor/model")

        self.segmentatorModel.to(device)
        self.extractorModel.to(device)
        self.abstractorModel.to(device)

    def __preprocess_segmentator(self, text):
        batched_encodings = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }
        for i in range(len(text) - self.context_window):
            batch = self.segmentatorTokenizer(
                text[i : i + self.context_window],
                padding=PaddingStrategy.LONGEST,
                truncation=True,
                return_tensors="pt",
                verbose=True,
            )
            for key in batched_encodings:
                batched_encodings[key].append(batch[key])

        return {
            "input_ids": torch.stack(batched_encodings["input_ids"]),
            "attention_mask": torch.stack(batched_encodings["attention_mask"]),
            "token_type_ids": torch.stack(batched_encodings["token_type_ids"]),
        }

    def __preprocess_extractor(self, text):
        batched_encodings = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
        }
        length = len(text) - self.context_window
        length = length if length > 0 else len(text)
        for i in range(length):
            batch = self.extractorTokenizer(
                text[i : i + self.context_window],
                padding=PaddingStrategy.LONGEST,
                truncation=True,
                return_tensors="pt",
                verbose=True,
            )
            for key in batched_encodings:
                batched_encodings[key].append(batch[key])

        return {
            "input_ids": torch.stack(batched_encodings["input_ids"]),
            "attention_mask": torch.stack(batched_encodings["attention_mask"]),
            "token_type_ids": torch.stack(batched_encodings["token_type_ids"]),
        }

    def __preprocess_abstractor(self, text):
        return self.abstractorTokenizer(
            text,
            padding=PaddingStrategy.LONGEST,
            truncation=True,
            return_tensors="pt",
            verbose=True,
        )

    def execute(self, text: list[str]):
        with torch.no_grad():
            summaries = []

            ################
            # SEGMENTATION #
            ################
            segmentatorData = self.__preprocess_segmentator(text)
            segmentatorOutput = self.segmentatorModel(**segmentatorData)
            segmentatorIndices = torch.where(segmentatorOutput[1] > 0.3)[0]
            segmentatorText = []
            start = 0
            for i in segmentatorIndices:
                end = i.item() + 1
                segmentatorText.append(text[start:end])
                start = end
            ##############
            # EXTRACTION #
            ##############
            for topic in segmentatorText:
                extractorData = self.__preprocess_extractor(topic)
                extractorOutput = self.extractorModel(**extractorData)
                extractorIndices = torch.where(extractorOutput[1] > 0.3)[0]
                extractorText = topic[extractorIndices]
                ###############
                # ABSTRACTION #
                ###############
                abstractorData = self.__preprocess_abstractor(extractorText)
                abstractorOutput = self.abstractorModel.generate(**abstractorData)
                summaries.append(
                    self.abstractorTokenizer.batch_decode(abstractorOutput)
                )

            return summaries
