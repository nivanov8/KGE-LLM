from embeddings.embedding_models.minilm import MiniLMV2EmbeddingModel
from data.fb15k_dataloaders import FB15k237DataModule

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


class ContrastiveCosineLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        """
        x: Tensor of shape (batch_size, dim)
        y: Tensor of shape (batch_size, dim)
        Returns: scalar loss
        """
        batch_size = x.size(0)

        # Normalize the embeddings
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(x, y.T)  # shape: (batch_size, batch_size)
        sim_matrix = sim_matrix / self.temperature

        # Create target labels: position i should match i
        targets = torch.arange(batch_size).to(x.device)

        # Cross-entropy loss between similarity scores and true indices
        loss_i = F.cross_entropy(sim_matrix, targets)
        loss_j = F.cross_entropy(sim_matrix.T, targets)

        # Average the two directions
        loss = (loss_i + loss_j) / 2
        return loss


# Hyperparams
batch_size = 4
num_epochs = 25
learning_rate = 3e-5


# Usage
csv_path = "/scratch/expires-2025-Apr-19/KGE/finetune_sam_copy.csv"
dataset = LLMOutputTailDataset(csv_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init model
model = MiniLMV2EmbeddingModel(device=device)
model.freeze_all_but_top_layers(num_layers_to_unfreeze=4)  # Finetune top 2 layers

# Loss and optimizer
#criterion = nn.CosineEmbeddingLoss(margin=0.2)
criterion = ContrastiveCosineLoss()
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

        # loss = criterion(llm_emb, tail_emb, target)
        loss = criterion(llm_emb, tail_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    # Save model after each epoch
    save_path = f"/scratch/expires-2025-Apr-19/KGE/finetuned"
    #torch.save(model.model.state_dict(), save_path)
    model.model.save_pretrained(save_path)
    print(f"Saved model checkpoint to {save_path}")

