from data.fb15k_dataloaders import FB15k237DataModule

## TEST
fb_loader = FB15k237DataModule(batch_size=16)
train_loader, valid_loader, test_loader = fb_loader.get_dataloaders()

for batch in test_loader:
    heads, relations, tails = batch
    for h, r, t in zip(heads, relations, tails):
        print(f"({h}, {r}, {t})")
    break