import os
import requests
from torch.utils.data import Dataset, DataLoader


# Download dataset
def download_file(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {url}")
        r = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(r.content)

base_url = "https://huggingface.co/datasets/KGraph/FB15k-237/resolve/main/data"


files_to_download = {
    "train.txt": "train.txt",
    "valid.txt": "valid.txt",
    "test.txt": "test.txt",
    "FB15k_mid2name.txt": "FB15k_mid2name.txt"
}

for filename, local_path in files_to_download.items():
    download_file(f"{base_url}/{filename}", local_path)

### Preprocessing

def load_mid2name(path):
    mid2name = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                mid2name[parts[0]] = parts[1]
    return mid2name

def load_triples(path, mid2name):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            h_name = mid2name.get(h, h)
            t_name = mid2name.get(t, t)
            triples.append((h_name, r, t_name))
    return triples

def split_relation_to_readable_list(relation):
    parts = relation.strip('/').replace('.', '/').split('/')
    return [p.replace('_', ' ') for p in parts if p]

def readable_relation_string(relation):
    """Return a cleaned, human-readable version of a relation string"""
    parts = relation.strip('/').replace('.', '/').split('/')
    return ', '.join(p.replace('_', '_') for p in parts if p)


## Build pytorch dataset

class ReadableTripleDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]

### Creating dataloader

mid2name = load_mid2name("FB15k_mid2name.txt")

train_triples = load_triples("train.txt", mid2name)
valid_triples = load_triples("valid.txt", mid2name)
test_triples  = load_triples("test.txt", mid2name)

train_loader = DataLoader(ReadableTripleDataset(train_triples), batch_size=32, shuffle=True)
valid_loader = DataLoader(ReadableTripleDataset(valid_triples), batch_size=32, shuffle=False)
test_loader  = DataLoader(ReadableTripleDataset(test_triples), batch_size=32, shuffle=False)


#### Test

for batch in test_loader:
    heads, relations, tails = batch
    for h, r, t in zip(heads, relations, tails):
        readable_r = readable_relation_string(r)
        #print(f"({h}, {r}, {t})") 
        print(f"({h}, {readable_r}, {t})")    
    break
