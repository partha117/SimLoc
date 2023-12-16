from typing import Optional, Dict
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config
from sentence_transformers import models

class CustomTransformer(models.Transformer):
    def __init__(self, without_weights: bool = False, **kwargs):
        self.without_weights = without_weights
        super(CustomTransformer, self).__init__(**kwargs)

    def _load_model(self, model_name_or_path, config, cache_dir):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir)
        else:
            if self.without_weights:
                self.auto_model = AutoModel.from_config(config=config)
            else:
                self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
