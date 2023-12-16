import torch
import wandb
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer, losses, InputExample, models

from report.pretrain.CustomTransformer import CustomTransformer

sys.path.append(str(Path(__file__).resolve().parents[1]))
import logging
from torch.nn import BCELoss, CosineEmbeddingLoss
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset, DatasetDict
import os
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from report.finetune.Models import DuplicateDetector, Residual
from report.pretrain.TSDAE_Training import create_arg_parser
from utils.Configuration import Configuration
from utils.Helper import get_metrics, setup_logger, get_pooler_class
import shutil


class LoggingCosineSimilarityLoss(losses.CosineSimilarityLoss):
    def forward(self, sentence_features, labels):
        loss_value = super().forward(sentence_features, labels)
        logging.info(f'Loss: {loss_value.item()}')
        run.log({"Loss": loss_value.item()})
        return loss_value


class FineTuneForDuplicateDetection:
    def __init__(self, config, save_path):
        self.config = config
        self.device = self.config.device
        if not self.config.pool:
            self.detector_model = SentenceTransformer(self.config.model_name_or_path, device=config.device)
        else:
            word_embedding_model = CustomTransformer(without_weights=False,
                                                     model_name_or_path=config.model_name_or_path)
            pooling_model = get_pooler_class(config.pool,
                                             word_embedding_dimension=word_embedding_model.get_word_embedding_dimension())
            modules = [word_embedding_model, pooling_model]
            if self. config.residual_layer > 0:
                self.residual_layers = nn.Sequential(
                    *[Residual(in_features=config.embedding_dimension, out_features=config.output_dimension) for _ in
                      range(self.residual_layers)])
                modules.append(self.residual_layers)
            self.detector_model = SentenceTransformer(modules=modules, device=config.device)
        self.save_path = os.path.join(save_path, self.config.id)
        self.metrics = [get_metrics(item) for item in self.config.metrics]

    def train_model(self, data_path):
        train_dataset = DatasetDict.load_from_disk(os.path.join(data_path, self.config.dataset))['train']
        train_dataset = train_dataset.map(lambda x: {"label": 1.0 if x['label'].lower() == 'yes' else 0.0, "texts": [x['bug1'], x['bug2']]})
        train_examples = [InputExample(texts=item['texts'], label=item['label']) for item in train_dataset]
        train_dataloader = DataLoader(train_examples, batch_size=self.config.batch_size, shuffle=True, num_workers=1)
        train_loss = LoggingCosineSimilarityLoss(self.detector_model)

        # Tune the model
        self.detector_model.fit(train_objectives=[(train_dataloader, train_loss)],
                                epochs=self.config.epochs,
                                scheduler=self.config.scheduler,#'warmuplinear',
                                warmup_steps=self.config.warmup_steps,#500,
                                optimizer_class=eval(self.config.optimizer),
                                weight_decay=self.config.weight_decay,
                                optimizer_params={'lr': self.config.learning_rate},
                                checkpoint_save_total_limit=self.config.save_limit,
                                checkpoint_save_steps=self.config.save_steps,
                                checkpoint_path=f"Output/DuplicateDetection/checkpoints/" if save_path is None else save_path,
                                show_progress_bar=True,
                                use_amp=False)

    def evaluate_model(self, data_path):
        test_dataset = DatasetDict.load_from_disk(os.path.join(data_path, self.config.dataset))['test']
        test_dataset = test_dataset.map(
            lambda x: {"label": 1.0 if x['label'].lower() == 'yes' else 0.0, "texts": [x['bug1'], x['bug2']]})
        similarity_method = nn.CosineSimilarity()
        embedding1 = self.detector_model.encode(test_dataset['bug1'], batch_size=self.config.batch_size, convert_to_tensor=True, show_progress_bar=True, device=self.config.device)
        embedding2 = self.detector_model.encode(test_dataset['bug2'], batch_size=self.config.batch_size,
                                                convert_to_tensor=True, show_progress_bar=True, device=self.config.device)
        predicted = similarity_method(embedding1, embedding2)
        actual = test_dataset['label']
        self.write_results(actual, predicted.cpu().numpy() > 0.8)

    def write_results(self, actual, predicted):
        Path(self.save_path).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(self.save_path, "results.txt"), "w") as f:
            for metric in self.metrics:
                performance = metric(actual, predicted)
                f.write(f"Metric: {metric.__name__} Performance: {performance}\n")
                print(f"Metric: {metric.__name__} Performance: {performance}\n")



if __name__ == "__main__":
    # path = "../../TrainingArgs/finetune/1.json"
    # config = Configuration(path=path)
    # save_path = f"/home/p9chakra/MY_DRIVES/ProgramFiles/SimLoc/Output/{config.id}"
    # data_path = "/home/p9chakra/MY_DRIVES/ProgramFiles/SimLoc/Data/Processed/"
    # dry_run = False

    parser = create_arg_parser("DuplicateDetection")
    args = parser.parse_args()
    path = args.config_path
    config = Configuration(path=path)
    save_path = args.save_path + f"{config.id}"
    data_path = args.data_path
    dry_run = args.dryRun

    setup_logger()
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project=os.path.basename(__file__),
        # Track hyperparameters and run metadata
        config=config,
    )
    trainer = FineTuneForDuplicateDetection(config=config, save_path=save_path)
    trainer.train_model(data_path=data_path)
    trainer.evaluate_model(data_path=data_path)
