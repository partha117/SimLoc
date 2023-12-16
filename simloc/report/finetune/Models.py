from torch import nn


class Residual(nn.Module):
    def __init__(self, in_features, out_features, norm_layer=None):
        super().__init__()
        self.dense = nn.Linear(in_features=in_features, out_features=out_features)
        self.act = nn.PReLU(in_features)
        if norm_layer == 'batch_norm':
            self.norm = nn.BatchNorm1d(in_features)
        elif norm_layer or norm_layer == 'layer_norm':
            self.norm = nn.LayerNorm(out_features)
        else:
            self.norm = None

    def forward(self, input):
        pre_act = self.dense(input)
        raw = self.act(pre_act) + input
        if self.norm:
            return self.norm(raw)
        return raw


class Projection(nn.Module):
    def __init__(self, in_features, out_features, norm_layer=None):
        super().__init__()
        self.dense = nn.Linear(in_features=in_features, out_features=out_features)
        self.act = nn.ReLU(in_features)

    def forward(self, input):
        pre_act = self.dense(input)
        return self.act(pre_act)


class DuplicateDetector(nn.Module):
    def __init__(self, embedding_model, residual_layers_count,output_dimension, pooling_method=None, embedding_dimension=None):
        super(DuplicateDetector, self).__init__()
        self.embedding_model = embedding_model
        self.residual = residual_layers_count
        self.pooling_method = pooling_method
        if self.residual > 0:
            self.residual_layers = nn.Sequential(
                *[Residual(in_features=embedding_dimension, out_features=output_dimension) for _ in
                  range(self.residual)])

    def _get_pooled_embeddings(self, embedding, attention):
        features = dict()
        features['token_embeddings'] = embedding
        features['attention_mask'] = attention
        return self.pooling_method.forward(features)['sentence_embedding']

    def forward(self, input_id1, input_id2, attention_mask1, attention_mask2):
        embeddings1 = self._get_pooled_embeddings(self.embedding_model(input_id1, attention_mask1).last_hidden_state,
                                                  attention_mask1)
        embeddings2 = self._get_pooled_embeddings(self.embedding_model(input_id2, attention_mask2).last_hidden_state,
                                                  attention_mask2)
        if self.residual > 0:
            embeddings1 = self.residual_layers(embeddings1)
            embeddings2 = self.residual_layers(embeddings2)
        return embeddings1, embeddings2
