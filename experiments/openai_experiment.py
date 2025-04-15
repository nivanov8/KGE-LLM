from data.fb15k_dataloaders import FB15k237DataModule
from tqdm import tqdm
from embeddings.embedding_models.modernBERT import ModernBERTEmbeddingModel
from embeddings.embedding_models.minilm import MiniLMV2EmbeddingModel
from embeddings.faiss.faiss_index import FreeBaseFaissIndex
from metrics.metrics import get_metrics
import torch
from openai import OpenAI
from itertools import islice

import torch.nn.functional as F

# Make sure to set your OpenAI API key
client = OpenAI(api_key="INSER")

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

def call_chatgpt(head, relation, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert knowledge graph completor. You will be given a head and a relation. Your task is to use that information and find a tail entity which is a result of the relation being applied to the head entity. Give your answer in as few words as possible.\n\n"},
            {"role": "user", "content": f"HEAD ENTITY: {head}\n\n RELATION: {relation}\n\nTAIL ENTITY:"}
        ],
        temperature=0.0,
        max_tokens=32
    )
    return response.choices[0].message.content.strip()

def run_expriment():
    fbloader = FB15k237DataModule(batch_size=1)
    #embedding_model = ModernBERTEmbeddingModel(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    #embedding_model = MiniLMV2EmbeddingModel(model_path="/scratch/expires-2025-Apr-19/svajpayee/checkpoints", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    #embedding_model = MiniLMV2EmbeddingModel(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    embedding_model = MiniLMV2EmbeddingModel(model_path="/scratch/expires-2025-Apr-19/KGE/finetuned", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    _, _, test_loader = fbloader.get_dataloaders()

    freebase_index = FreeBaseFaissIndex(embedding_model, "data/FB15k_mid2name.txt")
    freebase_index.load_index()
    entities = freebase_index._preprocess_entities()
    entities_id = [x[0] for x in entities]

    predictions, truths = [], []
    num_examples = 20
    for example in tqdm(islice(test_loader, num_examples), total=num_examples):
        head_id, head, relation, tail, tail_id = example
        head_id, head, relation, tail, tail_id = head_id[0], head[0], relation[0], tail[0], tail_id[0]

        #prompt = get_prompt(head, relation)

        
        out = call_chatgpt(head, relation)
        #out = "October"
        
        #print(head, relation, tail, out)
        ground_truth_idx = entities_id.index(tail_id)
        #print(freebase_index.faiss_index.reconstruct(ground_truth_idx))
        #print(embedding_model.get_embedding(["October"]))
        embedding = embedding_model.get_embedding([out]).detach().cpu()
        embedding_true = embedding_model.get_embedding([tail]).detach().cpu()
        #print(embedding_true)

        print(torch.inner(embedding, embedding_true).item())
        distances, indices = freebase_index.faiss_index.search(embedding, k=20)
        print(distances)

        indices = indices.tolist()[0]
        print([entities[i] for i in indices])

        ground_truth_idx = entities_id.index(tail_id)
        predictions.append(indices)
        truths.append(ground_truth_idx)

    metrics = get_metrics(predictions, truths)
    print(metrics)

if __name__ == "__main__":
    run_expriment()