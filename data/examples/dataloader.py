from data.fb15k_dataloaders import FB15k237DataModule

## TEST
fb_loader = FB15k237DataModule(batch_size=16)
train_loader, valid_loader, test_loader = fb_loader.get_dataloaders()

for batch in test_loader:
    head_id, heads, relations, tails, tail_id = batch
    for h_id, h, r, t, t_id in zip(head_id, heads, relations, tails, tail_id):
        print(f"({h_id}, {h}, {r}, {t}, {t_id})")
    break