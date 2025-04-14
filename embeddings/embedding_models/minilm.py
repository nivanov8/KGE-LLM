import torch
import os
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

from embeddings.embedding_type import EmbeddingType

os.environ["TRITON_CACHE_DIR"] = "/scratch/expires-2025-Apr-19/KGE"
torch.set_float32_matmul_precision('high')

class MiniLMV2EmbeddingModel():

    def __init__(self, cache_dir="/scratch/expires-2025-Apr-19/KGE", device=None):
        self.cache_dir = cache_dir
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = torch.device("cpu") if not device else device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
    

    def get_embedding(self, batch, embedding_type = EmbeddingType.CLS_TOKEN_EMBEDDING):
        inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
        self.model.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        batch_embeddings = self.mean_pooling(outputs, inputs["attention_mask"])

        return F.normalize(batch_embeddings, p=2, dim=1)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


