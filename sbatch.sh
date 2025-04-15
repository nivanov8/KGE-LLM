#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode24
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/expires-2025-Apr-10/svajpayee/bert_llm.out
#SBATCH --error=/scratch/expires-2025-Apr-10/svajpayee/bert_llm.err

srun python3 finetune/finetuned-minilm.py 