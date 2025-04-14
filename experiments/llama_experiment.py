from llama.llama import LlamaModel
from data.fb15k_dataloaders import FB15k237DataModule
from tqdm import tqdm
from embeddings.embedding_models.modernBERT import ModernBERTEmbeddingModel
from embeddings.faiss.faiss_index import FreeBaseFaissIndex
from metrics.metrics import get_metrics
import torch

from itertools import islice

def get_prompt(head, relation):
    return (
        "You are an expert knowledge graph completor. "
        "You will be given a head and a relation. "
        "Your task is to use that information and find a tail entity "
        "which is a result of the relation being applied to the head entity. "
        "Give your answer in as few words as possible.\n\n"
        f"HEAD ENTITY: {head}\n\n"
        f"RELATION: {relation}\n\n"
        "TAIL ENTITY:"
    )

def get_prompt_for_relation(relation):
    return (
        "You are an expert converter"
        "You will be given a relation in path format"
        "Your task is to use that convert the path to a human readible relation. "
        "Ensure to use information from the whole path\n\n"
        f"RELATION: {relation}\n\n"
        "HUMAN READIBLE RELATION: "
    )


def run_expriment():
    model = LlamaModel()
    fbloader = FB15k237DataModule(batch_size=1)
    modernBERT = ModernBERTEmbeddingModel(device=torch.device("cuda"))
    _, _, test_loader = fbloader.get_dataloaders()

    freebase_index = FreeBaseFaissIndex(modernBERT, "data/FB15k_mid2name.txt")
    freebase_index.load_index()
    entities = freebase_index._preprocess_entities()
    entities_id = [x[0] for x in entities]

    predictions, truths = [], []
    for example in tqdm(test_loader):
        head_id, head, relation, tail, tail_id = example
        head_id, head, relation, tail, tail_id = head_id[0], head[0], relation[0], tail[0], tail_id[0]

        prompt = get_prompt(head, relation)
        _, out = model.generate(prompt)

        embedding = modernBERT.get_embedding([out]).cpu()

        distances, indices = freebase_index.faiss_index.search(embedding, k=10)
        ground_truth_idx = entities_id.index(tail_id)
        
        predictions.append(indices.tolist())
        truths.append(ground_truth_idx)



    metrics = get_metrics(predictions, truths)
    print(metrics)


if __name__ == "__main__":
    run_expriment()