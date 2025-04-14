import os
import requests
from torch.utils.data import Dataset, DataLoader

class FB15k237DataModule:
    def __init__(self, data_dir='data/', batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.base_url = "https://huggingface.co/datasets/KGraph/FB15k-237/resolve/main/data"
        self.files = {
            "train.txt": os.path.join(data_dir, "train.txt"),
            "valid.txt": os.path.join(data_dir, "valid.txt"),
            "test.txt": os.path.join(data_dir, "test.txt"),
            "FB15k_mid2name.txt": os.path.join(data_dir, "FB15k_mid2name.txt")
        }
        self._download_dataset()
        self._load_mappings()
        self._prepare_dataloaders()

    def _download_dataset(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for filename, path in self.files.items():
            if not os.path.exists(path):
                print(f"Downloading {filename}...")
                r = requests.get(f"{self.base_url}/{filename}")
                with open(path, 'wb') as f:
                    f.write(r.content)

    def _load_mappings(self):
        self.mid2name = {}
        with open(self.files["FB15k_mid2name.txt"], "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    self.mid2name[parts[0]] = parts[1]

    @staticmethod
    def split_relation_no_underscore_cleanup(relation):
        parts = relation.strip('/').replace('.', '/').split('/')
        return [p for p in parts if p]

    def _load_triples(self, filepath):
        triples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                h, r, t = line.strip().split("\t")
                h_name = self.mid2name.get(h, h)
                t_name = self.mid2name.get(t, t)
                r_clean = r.replace('.', '/')  # just clean up dots, keep slashes
                triples.append((h_name, r_clean, t_name))
        return triples
    
    def _prepare_dataloaders(self):
        class ReadableTripleDataset(Dataset):
            def __init__(self, triples):
                self.triples = triples
            def __len__(self):
                return len(self.triples)
            def __getitem__(self, idx):
                return self.triples[idx]

        self.train_dataset = ReadableTripleDataset(self._load_triples(self.files["train.txt"]))
        self.valid_dataset = ReadableTripleDataset(self._load_triples(self.files["valid.txt"]))
        self.test_dataset  = ReadableTripleDataset(self._load_triples(self.files["test.txt"]))

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader  = DataLoader(self.test_dataset,  batch_size=self.batch_size, shuffle=False)

    def get_dataloaders(self):
        return self.train_loader, self.valid_loader, self.test_loader
