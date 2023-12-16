import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from sentence_transformers import SentenceTransformer
from sentence_transformers import models, datasets, losses
from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from utils.Configuration import Configuration
from report.pretrain.CustomTransformer import CustomTransformer
from utils.Helper import get_pooler_class, setup_logger
import wandb
import os
import argparse
import nltk
from report.pretrain.Pooler import *
def create_arg_parser(description="App"):
    # Creates and returns the ArgumentParser object

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config_path',
                    help='Path to the config.')
    parser.add_argument('--data_path',
                        help='Path to the config.')
    parser.add_argument('--save_path',
                    help='Path to the output ')
    parser.add_argument('--dryRun',action='store_true', required=False,
                        help='Is dryrun ')
    return parser
class LoggingDenoisingAutoEncoderLoss(losses.DenoisingAutoEncoderLoss):
    def forward(self, sentence_features, labels):
        loss_value = super().forward(sentence_features, labels)
        logging.info(f'Loss: {loss_value.item()}')
        run.log({"Loss": loss_value.item()})
        return loss_value
def train(model, model_name, train_data, train_args, save_path=None, filter_data=False):
    if filter_data:
        train_data = list(filter(lambda x: (len(x) * 0.25 <= 512), train_data))
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True)
    train_loss = LoggingDenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)

    # Call the fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=train_args.epochs,
        scheduler=train_args.scheduler,#'warmuplinear',
        warmup_steps=train_args.warmup_steps,#500,
        optimizer_class=eval(train_args.optimizer),
        weight_decay=train_args.weight_decay,
        optimizer_params={'lr': train_args.learning_rate},
        checkpoint_save_total_limit=5,
        checkpoint_path=f"Output/{model_name}/checkpoints/" if save_path is None else save_path,
        show_progress_bar=True,
        use_amp=False,
    )

    model.save(f'Output/{model_name}/tsdae-model' if save_path is None else save_path)

if __name__ == "__main__":
    # assert len(sys.argv) >= 2, "Needs two arguments"
    #
    # # print(os.getcwd())
    # path = "../../TrainingArgs/pretrain/7.json"
    # config = Configuration(path=path)
    # save_path = f"../../Output/{config.id}"
    # data_path = "../../../Data/Processed/"
    # dry_run = False

    parser = create_arg_parser("TSDAE")
    args = parser.parse_args()
    path = args.config_path
    config = Configuration(path=path)
    save_path = args.save_path + f"{config.id}"
    data_path = args.data_path
    dry_run = args.dryRun

    nltk.download('punkt')
    setup_logger()
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project=os.path.basename(__file__),
        # Track hyperparameters and run metadata
        config=config,
    )
    logging.info("Creating model.")
    word_embedding_model = CustomTransformer(without_weights=config.reset,model_name_or_path=config.model_name,tokenizer_args=config.tokenizer,)
    logging.info("Creating pooler.")
    pooling_model = get_pooler_class(config.pool,word_embedding_dimension=word_embedding_model.get_word_embedding_dimension())
    logging.info("Creating model and pooler.")
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=config.device)
    logging.info("Loading dataset")
    dataset = Dataset.load_from_disk(os.path.join(data_path, config.dataset)).shuffle()
    if dry_run:
        exit(0)
    logging.info("Starting training.")
    train(model=model, model_name=config.model_name, train_data=dataset['bug_report'], train_args=config, save_path=save_path, filter_data=True)
