import torch

from embeddings.faiss.faiss_index import FreeBaseFaissIndex
from embeddings.embedding_models.modernBERT import ModernBERTEmbeddingModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = ModernBERTEmbeddingModel(device=device)

index = FreeBaseFaissIndex(embedding_model=embedding_model, entities_data_path="data/FB15k_mid2name.txt")
index.add_embeddings_to_index()
index.save_index()

# index.load_index()