import sys
import logging
from pathlib import Path

from report.pretrain.TSDAE_Training import create_arg_parser

sys.path.append(str(Path(__file__).resolve().parents[1]))
from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset as TDataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers.data.data_collator import DataCollatorForWholeWordMask
from utils.Configuration import Configuration
import wandb
import os
import argparse
import nltk
import torch
from accelerate import Accelerator



class ReportDataset(TDataset):
    def __init__(self, tokenizer, raw_datasets, text_column_name: str, max_length: int, gradient_accumulation_steps=5):
        self.padding = "max_length"
        self.text_column_name = text_column_name
        self.max_length = max_length
        self.accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
        self.tokenizer = tokenizer

        with self.accelerator.main_process_first():
            self.tokenized_datasets = raw_datasets.map(
                self.tokenize_function,
                batched=True,
                num_proc=4,
                remove_columns=raw_datasets.column_names,
                desc="Running tokenizer on dataset line_by_line",
            ).shuffle()
            self.tokenized_datasets.set_format('torch', columns=['input_ids'], dtype=torch.long)

    def tokenize_function(self, examples):
        examples[self.text_column_name] = [
            line for line in examples[self.text_column_name] if len(line[0]) > 0 and not line[0].isspace()
        ]
        return self.tokenizer(
            examples[self.text_column_name],
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )

    def __len__(self):
        return len(self.tokenized_datasets)

    def __getitem__(self, i):
        return self.tokenized_datasets[i]


def train(model, train_data, train_args, mlm_prob, save_path=None):
    logging.info(f"MLM probability {mlm_prob}")
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
    tokenized_dataset_train = ReportDataset(
        tokenizer=tokenizer,
        text_column_name='bug_report',
        raw_datasets=train_data,
        max_length=train_args.tokenizer['max_length'],
    )
    training_args = TrainingArguments(
        output_dir=save_path,
        overwrite_output_dir=True,
        num_train_epochs=train_args.epochs,
        per_device_train_batch_size=train_args.batch_size,
        warmup_steps=train_args.warmup_steps,
        save_steps=100,
        logging_steps=10,
        weight_decay=train_args.weight_decay,
        lr_scheduler_type=train_args.scheduler,
        gradient_accumulation_steps=5,
        save_total_limit=5,
        save_safetensors=True,
        learning_rate=train_args.learning_rate,
        report_to='wandb',
        use_cpu=True if config.device == 'cpu' else False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset_train
    )
    trainer.train()


if __name__ == "__main__":
    # assert len(sys.argv) >= 2, "Needs two arguments"
    #

    # path = "../../TrainingArgs/pretrain/10.json"
    # config = Configuration(path=path)
    # save_path = f"../../Output/{config.id}"
    # data_path = "../../../Data/Processed/"
    # dry_run = False

    parser = create_arg_parser("MLM")
    args = parser.parse_args()
    path = args.config_path
    config = Configuration(path=path)
    save_path = args.save_path + f"{config.id}"
    data_path = args.data_path
    dry_run = args.dryRun

    nltk.download('punkt')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project=os.path.basename(__file__),
        # Track hyperparameters and run metadata
        config=config,
    )
    logging.info("Creating model.")
    if 'path' in config:
        model = AutoModelForMaskedLM.from_pretrained(config.path)
    else:
        model_config = AutoConfig.from_pretrained(config.model_name)
        model = AutoModelForMaskedLM.from_config(model_config)
    logging.info("Creating tokenizer.")
    if "tokenizer_path" in config:
        tokenizer_base = AutoTokenizer.from_pretrained(config.tokenizer_path)
    else:
        tokenizer_base = AutoTokenizer.from_pretrained(config.model_name)
    logging.info("Loading dataset")
    if config.dataset == "title_body":
        logging.info(f"Loading dataset {config.dataset}")
        dataset = DatasetDict.load_from_disk(os.path.join(data_path, config.dataset))["train"]
        dataset = dataset.map(lambda x: {"bug_report": x['title'] + " " + x['body']})
    else:
        dataset = Dataset.load_from_disk(os.path.join(data_path, config.dataset))
    if dry_run:
        exit(0)

    if not hasattr(config, "pretrain_tokenizer") or not config.use_pretrained_tokenizer:
        tokenizer = tokenizer_base.train_new_from_iterator(dataset['bug_report'], 50265,
                                                       show_progress=True)
    tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
    logging.info("Starting training.")
    train(model=model, train_data=dataset, train_args=config, mlm_prob=config.mlm_prob, save_path=save_path)
