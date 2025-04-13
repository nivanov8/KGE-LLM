import faiss

from embeddings.embedding_type import EmbeddingType


class FreeBaseFaissIndex():

    def __init__(self, embedding_model, entities_data_path="KGE-LLM/FB15k_mid2name.txt", embedding_type = EmbeddingType.MEAN_POOLING_EMBEDDING):
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        self.embedding_dim = self.embedding_model.model.config.hidden_size
        self.entities_data_path = entities_data_path

        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
    
    def add_embeddings_to_index(self):
        all_entities = self._preprocess_entities()
        
        # create embeddings for entities
        print(f"Starting to create embeddings, could take a minute")
        entity_embeddings = self.embedding_model.get_embedding(all_entities, self.embedding_type).cpu()
        print(f"Done creating embeddings.")

        # index the embeddings
        self.faiss_index.add(entity_embeddings)

        print(f"Written total of {self.faiss_index.ntotal} embeddings to index")
    
    def save_index(self, save_dir="/scratch/expires-2025-Apr-19/KGE/entities_index.faiss"):
        faiss.write_index(self.faiss_index, save_dir)
    
    def load_index(self, load_dir="/scratch/expires-2025-Apr-19/KGE/entities_index.faiss"):
        self.faiss_index = faiss.read_index(load_dir)
        print(f"Loaded faiss index: {self.faiss_index} with {self.faiss_index.ntotal} embeddings")
    
    def _preprocess_entities(self) -> list[str]:
        with open(self.entities_data_path, "r") as f:
            lines = f.readlines()
        
        entities = []
        for line in lines:
            _, name_part = line.strip().split(maxsplit=1)
            name_part = name_part.strip()
            name_part = name_part.replace("_", " ")
            entities.append(name_part)
    
        return sorted(entities)
