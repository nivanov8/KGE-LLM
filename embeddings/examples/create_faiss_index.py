import torch

from embeddings.faiss.faiss_index import FreeBaseFaissIndex
from embeddings.embedding_models.modernBERT import ModernBERTEmbeddingModel
from embeddings.embedding_models.minilm import MiniLMV2EmbeddingModel

from embeddings.embedding_type import EmbeddingType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#embedding_model = MiniLMV2EmbeddingModel(model_path="/scratch/expires-2025-Apr-19/svajpayee/checkpoints", device=device)
#embedding_model = MiniLMV2EmbeddingModel(device=device)
embedding_model = MiniLMV2EmbeddingModel(model_path="/scratch/expires-2025-Apr-19/KGE/finetuned", device=device)
#embedding_model = ModernBERTEmbeddingModel(device=device)

index = FreeBaseFaissIndex(embedding_model=embedding_model, entities_data_path="data/FB15k_mid2name.txt", embedding_type = EmbeddingType.MEAN_POOLING_EMBEDDING)
index.add_embeddings_to_index()
index.save_index()

# index.load_index()