"""
This script is designed to execute a fine-tuning job using the OpenAI API. It reads a configuration file to obtain the necessary parameters and submits a fine-tuning job with a specified dataset.

Key Components:
- `create_finetune`: Submits a fine-tuning job to the OpenAI API using the specified dataset and model, you can check the status of the job at https://platform.openai.com/finetune/ftjob-QPje3bZiOQ4shvQs6rKnRkyn.

Important Considerations:
1. Configuration File: Ensure that the `config_finetuning.yaml` file is correctly set up with the necessary parameters such as `dataset_finetune_name`, `model_name`, and `especial_suffix`.
2. Environment Variables: The `apis.env` file must contain valid OpenAI API credentials.
3. Dataset File: The dataset file should be located in the specified experiment directory and formatted correctly for fine-tuning.

Usage:
Run the script from the command line with the experiment path as an argument:
    python execute_finetune.py <EXPERIMENT_PATH>

Example:
    python execute_finetune.py my_experiment

This will submit a fine-tuning job using the specified dataset and model configuration.
"""

import sys

from utils import load_config, openai_login, read_yaml, huggingface_login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from datasets import DatasetDict


def create_finetune(file_path: str, model_name: str, suffix: str):
    file_object = client.files.create(file=open(file_path, "rb"), purpose="fine-tune")
    finetune_job = client.fine_tuning.jobs.create(
        training_file=file_object.id, model=model_name, suffix=suffix
    )
    return finetune_job

def finetune_open_weight(file_path: str, model_name: str, suffix: str):
    device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16)
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        task_type=TaskType.CAUSAL_LM,
        # target_modules=["q_proj", "v_proj", ...]  # optionally indicate target modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    dataset = load_dataset("json", data_files=file_path)["train"]

    dataset = dataset.train_test_split(test_size=0.1, seed=config_args['random_state'])
    dataset = DatasetDict({
        "train": dataset["train"],
        "validation": dataset["test"],
    })
    max_length = 2048
    
    def preprocess(example):
        conversation = ""
        for msg in example["messages"]:
            if msg["role"] == "user":
                conversation += "User: " + msg["content"] + "\n"
            elif msg["role"] == "assistant":
                conversation += "Assistant: " + msg["content"] + "\n"
    
        tokenized = tokenizer(conversation, truncation=True, max_length=max_length)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(preprocess, remove_columns=["messages"])
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    output_dir = f"{EXPERIMENT_PATH}/finetuning/{suffix}_lora"
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        gradient_accumulation_steps=4,
        bf16=True,
        report_to="none",
    )
    
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        EXPERIMENT_PATH = sys.argv[1]
        if len(sys.argv) > 2:
            FT_NAME = sys.argv[2]
    else:
        print(
            "Provide as arguments the experiment path and the fine-tuned model name, i.e.: python3 execute_finetune.py <EXPERIMENT_PATH> <FT_NAME>."
        )
        exit()

    config_args = load_config(
        experiment_path=EXPERIMENT_PATH,
        config_type="finetuning",
        name=FT_NAME,
    )

    company = config_args.get("company")
    if company == "OpenAI":
        client = openai_login()
    
        finetune_job = create_finetune(
            file_path=f"{EXPERIMENT_PATH}/finetuning/{config_args['dataset_finetune_name']}",
            model_name=config_args["model_name"],
            suffix=config_args["especial_suffix"],
        )
        # https://platform.openai.com/finetune/ftjob-QPje3bZiOQ4shvQs6rKnRkyn
    elif company == "HuggingFace":
        huggingface_login()

        finetune_open_weight(
            file_path=f"{EXPERIMENT_PATH}/finetuning/{config_args['dataset_finetune_name']}",
            model_name=config_args["model_name"],
            suffix=config_args["especial_suffix"],
        )
    elif company == "Local":
        finetune_open_weight(
            file_path=f"{EXPERIMENT_PATH}/finetuning/{config_args['dataset_finetune_name']}",
            model_name=config_args["model_name"],
            suffix=config_args["especial_suffix"],
        )
    else:
        print(f"Company {company} is not supported.")
