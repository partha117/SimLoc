import torch
import wandb
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
import os
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from report.finetune.Models import DuplicateDetector
from report.pretrain.TSDAE_Training import create_arg_parser
from utils.Configuration import Configuration
from utils.Helper import get_metrics, setup_logger


class FineTuneForDuplicateDetection:
    def __init__(self, config, save_path):
        self.config = config
        self.device = config.device
        self.model = AutoModel.from_pretrained(config.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.detector_model = DuplicateDetector(embedding_model=self.model,
                                                embedding_dimension=config.embedding_dimension,
                                                output_dimension=config.output_dimension,
                                                residual_layers_count=0).to(self.device)
        self.activator = nn.Sigmoid()
        self.optimizer = eval(config.optimizer)(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = eval(config.scheduler)(self.optimizer, factor=config.weight_decay)
        self.criterion = eval(config.criterion)
        self.save_path = os.path.join(save_path, config.id)
        self.metrics = [get_metrics(item) for item in config.metrics]

    def train_model(self, data_path, config):
        train_dataset = Dataset.load_from_disk(os.path.join(data_path, config.dataset))
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        for epoch in range(config.epochs):
            for batch in train_loader:
                input_ids1, attention_mask1 = self.tokenizer(batch['bug1'])
                input_ids2, attention_mask2 = self.tokenizer(batch['bug2'])
                label = batch['label']
                prediction = self.detector_model(input_ids1.to(self.device), input_ids2.to(self.device), attention_mask1.to(self.device), attention_mask2.to(self.device))
                loss = self.criterion(self.activator(prediction), label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

    def evaluate_model(self, data_path, config):
        test_dataset = Dataset.load_from_disk(os.path.join(data_path, config.dataset))
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        actual_value = []
        predicted_value = []
        for batch in test_loader:
            input_ids1, attention_mask1 = self.tokenizer(batch['bug1'])
            input_ids2, attention_mask2 = self.tokenizer(batch['bug2'])
            label = batch['label']
            with torch.no_grad():
                prediction = self.activator(self.detector_model(input_ids1.to(self.device), input_ids2.to(self.device), attention_mask1.to(self.device), attention_mask2.to(self.device)))
            actual_value.append(label)
            predicted_value.extend(prediction.numpy().tolist())
        self.write_results(actual_value, predicted_value)

    def write_results(self, actual, predicted):
        with open(os.path.join(self.save_path, "results.txt"), "w") as f:
            for metric in self.metrics:
                performance = metric(actual, predicted)
                f.write(f"Metric: {metric} Performance: {performance}\n")
                print(f"Metric: {metric} Performance: {performance}\n")
        self.model.save_pretrained(os.path.join(self.save_path, "hf_model.pt"))
        torch.save(self.detector_model.state_dict(), os.path.join(self.save_path, "model.pt"))


if __name__ == "__main__":
    # path = "../TrainingArgs/pretrain/2.json"
    # config = Configuration(path=path)
    # save_path = f"../../Output/{config.id}"
    # data_path = "../../Data/Processed/"
    # dry_run = False

    parser = create_arg_parser("TSDAE")
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


