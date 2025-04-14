import tiktoken
from data.fb15k_dataloaders import FB15k237DataModule

enc = tiktoken.encoding_for_model("gpt-4o")
test_token_count = 0


def format_prompt(head, relation):
    return [
        {
            "role": "system",
            "content": (
                "You are an expert knowledge graph completor. You will be given a head and a relation. "
                "Your task is to use that information and find a tail entity which is a result of the relation "
                "being applied to the head entity. Give your answer in as few words as possible."
            )
        },
        {
            "role": "user",
            "content": f"HEAD ENTITY: {head}\n\nRELATION: {relation}\n\nTAIL ENTITY:"
        }
    ]

def format_string_prompt(prompt):
    return '\n'.join([f"{p['role'].upper()}: {p['content']}" for p in prompt])



fb_loader = FB15k237DataModule(batch_size=1)  # batch size 1 for token-level granularity
train_loader, _, test_loader = fb_loader.get_dataloaders()
ctr = 0
for batch in train_loader:
    ctr += 1
    if ctr < 10000:
        h_id, heads, relations,_, _ = batch
        for h, r in zip(heads, relations):
            prompt = format_prompt(h, r)
            string_prompt = format_string_prompt(prompt)
            test_token_count += len(enc.encode(string_prompt))  # len of encoded token list
    else:
        break

print(f"Total tokens used in test prompts: {test_token_count}")