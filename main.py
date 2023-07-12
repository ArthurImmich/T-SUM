import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import nltk
import numpy as np
import torch
import transformers
from AMICorpusHandler import AMICorpusHandler
from datasets import DatasetDict
from evaluate import load
from ExtractorModel import ExtractorModel
from ExtractorTrainer import ExtractorTrainer
from filelock import FileLock
from nltk.metrics.segmentation import ghd, pk, windowdiff
from sklearn.metrics import f1_score, precision_recall_curve
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DefaultDataCollator,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.file_utils import is_offline_mode
from transformers.tokenization_utils import PaddingStrategy
from transformers.trainer_utils import get_last_checkpoint, is_main_process

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


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default="./cache",
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )


@dataclass
class DataTrainingArguments:
    ami_xml_dir: str = field(
        default="./data",
        metadata={"help": "AMI Corpus download directory"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=2,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    context_window: Optional[int] = field(
        default=9,
        metadata={"help": "The size of context window to be used by cross attention."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this value if set."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )


def load_dataset(ami_xml_dir, dataset_cache_file):
    if os.path.isdir(dataset_cache_file):
        return DatasetDict.load_from_disk(dataset_cache_file)
    datasets = AMICorpusHandler(ami_corpus_dir=ami_xml_dir).get_all_meetings_data()
    datasets = datasets.train_test_split(test_size=0.30, shuffle=False)
    aux_datasets = datasets["test"].train_test_split(test_size=0.5, shuffle=False)
    datasets["test"] = aux_datasets["train"]
    datasets["validation"] = aux_datasets["test"]
    datasets.save_to_disk(dataset_cache_file)
    return datasets


def parse_arguments():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    return parser.parse_args_into_dataclasses()


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def compute_metrics(eval_pred):
    metrics = ["accuracy", "recall", "precision", "f1"]
    metric = {}
    for met in metrics:
        metric[met] = load(met)
    logits = eval_pred.predictions.flatten()
    labels = eval_pred.label_ids.flatten()
    logits = 1 / (1 + np.exp(-np.array(logits)))

    _, _, thresholds = precision_recall_curve(labels, logits)
    f1_scores = [f1_score(labels, logits >= t) for t in thresholds]
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    print("==============================")
    print(f"Optimal Treshold: {optimal_threshold}")
    print("==============================")

    predictions = (logits >= optimal_threshold).astype(int)

    metric_res = {}
    for met in metrics:
        metric_res[met] = metric[met].compute(
            predictions=predictions, references=labels
        )[met]
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    # window_size = int(len(labels) * 0.25)
    # predictions = "".join([str(x) for x in predictions])
    # labels = "".join([str(x) for x in labels])
    # metric_res["pk"] = pk(ref=predictions, hyp=labels, k=window_size)
    # metric_res["windowdiff"] = windowdiff(seg1=predictions, seg2=labels, k=window_size)
    # metric_res["ghd"] = ghd(ref=predictions, hyp=labels)
    return metric_res


def preprocess_dataset(data, tokenizer, max_length=512, context_window=9):
    context_batches = {"sentences": [], "extractive": []}
    for sentences, extractive in zip(data["sentences"], data["extractive"]):
        for i in range(len(sentences) - (context_window - 1)):
            context_batches["sentences"].append(
                tokenizer(
                    sentences[i : i + context_window],
                    add_special_tokens=True,
                    padding=PaddingStrategy.MAX_LENGTH,
                    truncation=True,
                    max_length=max_length,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                    verbose=True,
                )
            )
            context_batches["extractive"].append(extractive[i : i + context_window])

    data = {
        "input_ids": torch.stack(
            [context["input_ids"] for context in context_batches["sentences"]]
        ),
        "token_type_ids": torch.stack(
            [context["token_type_ids"] for context in context_batches["sentences"]]
        ),
        "attention_mask": torch.stack(
            [context["attention_mask"] for context in context_batches["sentences"]]
        ),
        "labels": torch.stack(context_batches["extractive"]),
    }

    return data


def main():
    model_args, data_args, training_args = parse_arguments()

    last_checkpoint = None

    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        num_labels=1,
        classifier_dropout=0.5,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    datasets = load_dataset(
        data_args.ami_xml_dir,
        f"{data_args.ami_xml_dir}cache/dataset",
    )

    preprocessed_dir = f"{data_args.ami_xml_dir}cache/preprocessed"
    if os.path.isdir(preprocessed_dir):
        datasets = DatasetDict.load_from_disk(preprocessed_dir)

    else:
        datasets = load_dataset(
            data_args.ami_xml_dir,
            f"{data_args.ami_xml_dir}cache/dataset",
        )

        datasets = datasets.map(
            preprocess_dataset,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": data_args.max_source_length,
            },
            batched=True,
            batch_size=5,
            remove_columns=datasets["train"].column_names,
        )

        datasets.save_to_disk(preprocessed_dir)

    if not (
        training_args.do_train or training_args.do_eval or training_args.do_predict
    ):
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    bce_pos_weight = None

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # Undersample the dataset
        train_dataset = train_dataset.filter(lambda x: sum(x["labels"]) > 0)

        train_dataset = train_dataset.shuffle()

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

        labels = train_dataset["labels"].flatten()
        pos_labels = sum(labels).item()
        bce_pos_weight = (len(labels) - pos_labels) / pos_labels

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    data_collator = DefaultDataCollator()

    model = ExtractorModel(
        config,
        bce_pos_weight=bce_pos_weight,
        sequence_length=data_args.max_source_length,
    )

    trainer = ExtractorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(
            resume_from_checkpoint=checkpoint,
            ignore_keys_for_eval=[
                "loss",
                "last_hidden_state",
                "pooler_output",
                "hidden_states",
                "past_key_values",
                "attentions",
                "cross_attentions",
            ],
        )
        trainer.save_model()

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )

        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            ignore_keys=[
                "loss",
                "last_hidden_state",
                "pooler_output",
                "hidden_states",
                "past_key_values",
                "attentions",
                "cross_attentions",
            ],
        )
        max_val_samples = (
            data_args.max_val_samples
            if data_args.max_val_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            ignore_keys=[
                "loss",
                "last_hidden_state",
                "pooler_output",
                "hidden_states",
                "past_key_values",
                "attentions",
                "cross_attentions",
            ],
        )
        metrics = test_results.metrics
        max_test_samples = (
            data_args.max_test_samples
            if data_args.max_test_samples is not None
            else len(test_dataset)
        )
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    return results


if __name__ == "__main__":
    main()
