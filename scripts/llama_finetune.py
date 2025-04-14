from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from dotenv import load_dotenv
from huggingface_hub import login
import torch
import os

class LlamaModel:
    def __init__(self, use_lora=True):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        login(token=hf_token)

        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        cache_dir = "/scratch/expires-2025-Apr-19"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)

        if use_lora:
            # Load in 4-bit and prepare for PEFT
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=cache_dir
            )
            self.model.config.use_cache = False
            self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=cache_dir
            )

    def generate(self, input):
        prompt = f"<|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **tokens,
                max_length=100,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        raw_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        processed_response = raw_response.split("assistant")[-1].strip()
        return raw_response, processed_response
