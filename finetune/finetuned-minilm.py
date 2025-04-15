from embeddings.embedding_models.minilm import MiniLMV2EmbeddingModel
from data.fb15k_dataloaders import FB15k237DataModule

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch

import pandas as pd
from torch.utils.data import Dataset, DataLoader

class LLMOutputTailDataset(Dataset):
    def __init__(self, csv_path):
        self.llm_outputs = []
        self.tail_entities = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')

                if len(parts) < 6:
                    continue  # skip malformed lines

                tail = parts[-3].strip()
                llm_output = parts[-1].strip()

                self.llm_outputs.append(llm_output)
                self.tail_entities.append(tail)

        # Construct prompts as LLM inputs
        # self.llm_outputs = df["llm_output"].tolist()
        # self.tail_entities = df["tail"].tolist()

    def __len__(self):
        return len(self.llm_outputs)

    def __getitem__(self, idx):
        return self.llm_outputs[idx], self.tail_entities[idx]


# Hyperparams
batch_size = 8
num_epochs = 5
learning_rate = 2e-5


# Usage
csv_path = "/scratch/expires-2025-Apr-19/KGE/finetune_sam_copy.csv"
dataset = LLMOutputTailDataset(csv_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Sample data (replace with actual dataset)
# llm_outputs = ["What currency is used in Lycoming County?", "Who is the president of Canada?"]
# tail_entities = ["United_States_dollar", "Justin_Trudeau"]


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init model
model = MiniLMV2EmbeddingModel(device=device)
model.freeze_all_but_top_layers(num_layers_to_unfreeze=4)  # Finetune top 2 layers

# Loss and optimizer
criterion = nn.CosineEmbeddingLoss(margin=0.2)
optimizer = optim.AdamW(model.model.parameters(), lr=learning_rate)

# Finetuning loop
model.model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for tail_batch, llm_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        llm_emb = model.get_embedding(llm_batch, training=True)
        tail_emb = model.get_embedding(tail_batch, training=True)

        # Target: 1 for matching pairs
        target = torch.ones(llm_emb.size(0)).to(device)

        loss = criterion(llm_emb, tail_emb, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    # Save model after each epoch
    save_path = f"/w/331/svajpayee/KGE-LLM/checkpoints/minilm_finetuned_epoch{epoch+1}.pt"
    torch.save(model.model.state_dict(), save_path)
    print(f"Saved model checkpoint to {save_path}")

