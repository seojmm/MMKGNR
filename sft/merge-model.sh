#!/bin/bash

# Base model (pretrained, not fine-tuned)
base_model_name_or_path="meta-llama/Llama-2-7b-hf"

# PEFT checkpoint path (e.g., sft/checkpoints/meta-llama-2-7b-hf/checkpoint-4000)
peft_checkpoint=$1

# Extract folder names
base_model_last_name=$(basename $base_model_name_or_path)
peft_checkpoint_directory=$(dirname $peft_checkpoint)
peft_checkpoint_step=$(basename $peft_checkpoint)
peft_run_name=$(basename $peft_checkpoint_directory)

# Save to: merged_checkpoints/meta-llama-2-7b-hf/meta-llama-2-7b-hf-4000
model_name_to_be_saved="merged_checkpoints/${base_model_last_name}/${peft_run_name}-${peft_checkpoint_step}"

echo "🔧 Merging LoRA adapter into base model..."
echo "📦 Base model      : $base_model_name_or_path"
echo "🧩 PEFT checkpoint : $peft_checkpoint"
echo "💾 Save path       : $model_name_to_be_saved"

python sft/merge_peft_adapters.py \
    --base_model_name_or_path $base_model_name_or_path \
    --peft_model_path $peft_checkpoint \
    --save_path $model_name_to_be_saved
