# KGE-LLM

## Environment setup
1. Ensure miniconda3 is installed
2. Source conda with ```source $YOUR_PATH/miniconda3/bin/activate```
3. If you are creating the environment for the first time, navigate to the KGE-LLM root directory run ```conda env create -f environment.yml```, if the environment is already created go to step 4
4. Run ```conda activate KGE```

## Hugging Face Access Token
1. To be able to run Llama models create a local .env file in the root directory of KGE-LLM
2. In the .env file create a variable ```HF_TOKEN = $YOUR_HF_TOKEN```

## Experiments
1. To run experiments run ```python -m experiments.openai_experiment```
2. Ensure the conda environment is active
3. Ensure you have an OpenAI API Key entered in the code

## Finetuning
1. To run finetuning run ```python -m finetune.finetuned-minilm
