import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Optional
import nltk
import numpy as np
import transformers
from AMICorpusHandler import AMICorpusHandler
from datasets import DatasetDict
from evaluate import load
from filelock import FileLock
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, HfArgumentParser
from transformers.file_utils import is_offline_mode
from transformers.models.bigbird_pegasus import BigBirdPegasusForConditionalGeneration
from transformers.tokenization_utils import PaddingStrategy
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from pandas import DataFrame

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def load_dataset(ami_xml_dir, dataset_cache_file):
    datasets = AMICorpusHandler(ami_corpus_dir=ami_xml_dir).get_all_meetings_data()
    datasets = datasets.train_test_split(test_size=0.30, shuffle=False)
    aux_datasets = datasets["test"].train_test_split(test_size=0.5, shuffle=False)
    datasets["test"] = aux_datasets["train"]
    datasets["validation"] = aux_datasets["test"]
    datasets.save_to_disk(dataset_cache_file)
    return datasets


def preprocess_dataset(data):
    for meeting in data["abstractive"]:
        abstract = [topic["abstract"] for topic in meeting]
        extract = [" ".join(topic["extract"]) for topic in meeting]

    data = {"text": extract, "target": abstract}

    return data


def main():
    datasets = load_dataset(
        "./ami",
        "./cache",
    )

    datasets = datasets.map(
        preprocess_dataset,
        batched=True,
        batch_size=5,
        remove_columns=datasets["train"].column_names,
    )

    columns = ["text", "target"]
    datasets.set_format(type="torch", columns=columns)
