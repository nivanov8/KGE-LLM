from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import os

class LlamaModel:
    def __init__(self):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        login(token=hf_token)

        cache_dir = "/scratch/expires-2025-Apr-19"

        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", cache_dir=cache_dir)

    def generate(self, input):
        prompt = f"<|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
        tokens = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(**tokens, max_length=300, eos_token_id=self.tokenizer.eos_token_id)

        raw_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        processed_response = raw_response.split("assistant")[-1].strip()

        return raw_response, processed_response
