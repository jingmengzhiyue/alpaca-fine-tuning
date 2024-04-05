import os
import sys
from typing import List

import fire  # CLI library for building command line interfaces
import torch
import transformers
from datasets import (
    load_dataset,
)  # Utility for loading and processing datasets


# Import specific utilities for LoRA (Low-Rank Adaptation) and model preparation

from peft import (
    LoraConfig,  # Configuration class for LoRA parameters
    get_peft_model,  # Function to apply LoRA modifications to a model
    get_peft_model_state_dict,  # Retrieves the state dict of a model with LoRA modifications
    prepare_model_for_kbit_training,  # Retrieves the state dict of a model with LoRA modifications
    set_peft_model_state_dict,  # Sets the state dict of a model with LoRA modifications
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter


def train(
    # model/data params
    base_model: str = "",  # the only required argument; Base model identifier for pretraining
    data_path: str = "yahma/alpaca-cleaned",  # Path to training dataset
    output_dir: str = "./lora-alpaca",  # Directory to save trained model and outputs
    # training hyperparams
    batch_size: int = 128,  # Total batch size for training
    micro_batch_size: int = 4,  # Batch size for each gradient accumulation step
    num_epochs: int = 3,  # Number of epochs to train for
    learning_rate: float = 3e-4,  # Learning rate for optimizer
    cutoff_len: int = 256,  # Maximum sequence length for model inputs
    val_set_size: int = 2000,  # Size of the validation set
    # lora hyperparams
    lora_r: int = 8,  # Rank of LoRA adjustments
    lora_alpha: int = 16,  # Scale of LoRA adjustments
    lora_dropout: float = 0.05,  # Dropout rate for LoRA layers
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],  # Transformer modules to apply LoRA
    # llm hyperparams
    train_on_inputs: bool = True,  # Flag to determine if model trains on inputs or masks them out, if False, masks out inputs in loss
    add_eos_token: bool = False,  # Whether to add an EOS token to each input sequence
    group_by_length: bool = False,  # Whether to group training data by length for efficiency. faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",  # Project name for Weights & Biases tracking
    wandb_run_name: str = "",  # Run name for Weights & Biases tracking
    wandb_watch: str = "",  # Level of model watching by Weights & Biases ('gradients', 'all', or 'false')
    wandb_log_model: str = "",  # Flag to log model to Weights & Biases ('true' or 'false')
    resume_from_checkpoint: str = None,  # Path to checkpoint to resume training from
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    # Prints training parameters if executed by the main process in a distributed setting
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    # Ensures that a base model is specified for training
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    # Calculates the number of steps for gradient accumulation based on batch sizes
    gradient_accumulation_steps = batch_size // micro_batch_size
    # Initializes the prompter with the specified template
    prompter = Prompter(prompt_template_name)

    # Configuration for device mapping and Distributed Data Parallel (DDP) setup
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    # Configuration for Weights & Biases integration based on environment variables and arguments
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    # Load the model and tokenizer with specified configurations
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    # Set tokenizer configurations for padding
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        """
        Functions for tokenizing prompts and data points are defined here
        These include adjustments for EOS tokens and handling training without input tokens
        """
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    # Model preparation for k-bit training and LoRA modifications
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # Loading the dataset from the specified path
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # Loading model weights from checkpoint if specified
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # Prints the percentage of trainable parameters for transparency
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # Splits the dataset into training and validation sets if specified
    if val_set_size > 0:
        # Dataset split and mapping to tokenization
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        # Only training dataset mapping to tokenization
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # Configuration for multi-GPU training without using Distributed Data Parallel
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # Initializes the Hugging Face Trainer with the specified training arguments and data collator
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    # Starts the training process, with option to resume from a checkpoint
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Saves the trained model
    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
