import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_peft_adapter(base_model_name_or_path, peft_model_path, save_path):
    print("🔄 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)

    print("🔄 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, peft_model_path)

    print("🔗 Merging adapter with base model...")
    model = model.merge_and_unload()

    print(f"💾 Saving merged model to: {save_path}")
    model.save_pretrained(save_path)

    print("✅ Done. Optionally saving tokenizer as well...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--peft_model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    args = parser.parse_args()
    merge_peft_adapter(
        base_model_name_or_path=args.base_model_name_or_path,
        peft_model_path=args.peft_model_path,
        save_path=args.save_path
    )
