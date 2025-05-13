import os
import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import bitsandbytes as bnb


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    use_auth: bool = field(default=False)

@dataclass
class DataArguments:
    dataset: str = field(default="../supervision/250512_101010.jsonl")
    source_max_len: int = field(default=2048)
    target_max_len: int = field(default=2048)
    dataset_format: str = field(default="input-output")
    eval_dataset_size: int = field(default=1000)
    max_eval_samples: Optional[int] = field(default=None)

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    output_dir: str = field(default="./output")
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    learning_rate: float = field(default=2e-4)
    max_steps: int = field(default=4000)
    bf16: bool = field(default=True)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=100)
    save_total_limit: int = field(default=50)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="constant")
    gradient_checkpointing: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="wandb")


class DataCollatorForCausalLM:
    def __init__(self, tokenizer, source_max_len, target_max_len, train_on_source=False):
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.train_on_source = train_on_source
        self.IGNORE_INDEX = -100

    def __call__(self, instances):
        input_ids, labels = [], []
        for instance in instances:
            source = self.tokenizer(
                instance["input"],
                truncation=True,
                max_length=self.source_max_len,
                add_special_tokens=False,
            )["input_ids"]
            target = self.tokenizer(
                instance["output"],
                truncation=True,
                max_length=self.target_max_len,
                add_special_tokens=False,
            )["input_ids"]

            input_ids.append(torch.tensor(source + target))
            if not self.train_on_source:
                labels.append(torch.tensor([self.IGNORE_INDEX] * len(source) + target))
            else:
                labels.append(torch.tensor(source + target))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.IGNORE_INDEX)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }


@dataclass
class LoRAArguments:
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=4)
    lora_modules: str = field(default="all")
    

def preprocess(samples):
    inputs, outputs = [], []
    for title, triples in zip(samples["title"], samples["triples"]):
        prompt = f'You are a triple extractor for constructing user behavior knowledge graphs with factual triples from news titles. I will give you "News title", "Image-guided triples". You should extract 10 triples from title. The triples should serve as a knowledge context for personalized news recommendation. ## Caution1 : Each triple is separated with "\\n" and must be provided in "head | relation | tail" format. ## News title: {title} ## Image-guided triples:'
        
        response = "\n".join(triples)
        inputs.append(prompt)
        outputs.append(response)
    return {"input": inputs, "output": outputs}


def make_data_module(tokenizer, data_args, training_args):
    dataset = load_dataset("json", data_files=data_args.dataset)["train"]
    dataset = dataset.map(preprocess)

    if training_args.do_eval:
        print("Splitting train dataset into train and validation...")
        dataset = dataset.train_test_split(
            test_size=data_args.eval_dataset_size / len(dataset), seed=42, shuffle=True
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        if training_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), training_args.max_eval_samples)))
    else:
        train_dataset = dataset
        eval_dataset = None

    if training_args.group_by_length:
        train_dataset = train_dataset.map(lambda x: {"length": len(x["input"]) + len(x["output"])})
        if eval_dataset:
            eval_dataset = eval_dataset.map(lambda x: {"length": len(x["input"]) + len(x["output"])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=data_args.source_max_len,
        target_max_len=data_args.target_max_len
    )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator
    }
    


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoRAArguments))
    model_args, data_args, training_args, lora_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_auth_token=model_args.use_auth)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        bnb_4bit_use_double_quant=lora_args.double_quant,
        bnb_4bit_quant_type=lora_args.quant_type
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=model_args.use_auth
    )
    model = prepare_model_for_kbit_training(model)

    target_modules = ["q_proj", "v_proj"] if lora_args.lora_modules == "all" else lora_args.lora_modules.split(",")
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    data_module = make_data_module(tokenizer, data_args, training_args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"]
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
