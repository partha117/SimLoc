from torch import Tensor
from typing import Dict
from sentence_transformers import models
import torch


class AnglePooling(models.Pooling):
    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        embedding = (token_embeddings[:, 0] + torch.mean(token_embeddings, dim=1)) / 2.0
        features.update({'sentence_embedding': embedding})
        return features


class ClsPooling(models.Pooling):
    def __init__(self, word_embedding_dimension: int):
        super(ClsPooling, self).__init__(word_embedding_dimension=word_embedding_dimension, pooling_mode='cls')


class MeanPooling(models.Pooling):
    def __init__(self, word_embedding_dimension: int):
        super(MeanPooling, self).__init__(word_embedding_dimension=word_embedding_dimension, pooling_mode='mean')


class MaxPooling(models.Pooling):
    def __init__(self, word_embedding_dimension: int):
        super(MaxPooling, self).__init__(word_embedding_dimension=word_embedding_dimension, pooling_mode='max')
