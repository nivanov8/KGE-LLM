from data.fb15k_dataloaders import FB15k237DataModule
from itertools import islice
from tqdm import tqdm
from openai import OpenAI
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


client = OpenAI(api_key="INSERT")

fb_loader = FB15k237DataModule(batch_size=1)  # batch size 1 for token-level granularity
train_loader, _, _ = fb_loader.get_dataloaders()

file_location = "/scratch/expires-2025-Apr-19/KGE/finetune.csv"
num_examples = 10000

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
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

with open(file_location, "a") as f:
    for batch in tqdm(islice(train_loader, num_examples), total=num_examples):
        head_ids, heads, relations, tails, tail_ids = batch
        head_id, head, relation, tail, tail_id = head_ids[0], heads[0], relations[0], tails[0], tail_ids[0]
        
        model_output = call_chatgpt(head, relation)
        time.sleep(1)

        f.write(f"{head_id},{head},{relation},{tail},{tail_id},{model_output}\n")
        f.flush()
