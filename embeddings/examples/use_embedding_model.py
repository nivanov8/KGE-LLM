import torch

from embeddings.embedding_models.modernBERT import ModernBERTEmbeddingModel
from embeddings.embedding_type import EmbeddingType


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = ModernBERTEmbeddingModel(device=device)

fake_entities = ["Winnie the Pooh", "Lionel Messi", "Cristiano Ronaldo"]

cls_embedding = embedding_model.get_embedding(fake_entities)
mean_pooling_embedding = embedding_model.get_embedding(fake_entities, embedding_type=EmbeddingType.MEAN_POOLING_EMBEDDING)

print(cls_embedding.shape)
print(mean_pooling_embedding.shape)
print(embedding_model.model.config.hidden_size)