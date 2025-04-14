import torch
import os
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

from embeddings.embedding_type import EmbeddingType

os.environ["TRITON_CACHE_DIR"] = "/scratch/expires-2025-Apr-19/KGE"
torch.set_float32_matmul_precision('high')

class ModernBERTEmbeddingModel():

    def __init__(self, cache_dir="/scratch/expires-2025-Apr-19/KGE", device=None):
        self.cache_dir = cache_dir
        self.model_name = "answerdotai/ModernBERT-base"
        self.device = torch.device("cpu") if not device else device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)

    def get_embedding(self, batch, embedding_type = EmbeddingType.CLS_TOKEN_EMBEDDING):
        inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
        self.model.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.last_hidden_state
        
        if embedding_type == EmbeddingType.CLS_TOKEN_EMBEDDING:
            return F.normalize(predictions[:, 0, :], p=2, dim=1)
        
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        mean_embedding = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)

        return F.normalize(mean_embedding, p=2, dim=1)


