# tune.unsloth.alpaca

```python

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt
from unsloth import apply_chat_template
from unsloth.chat_templates import train_on_responses_only

import torch
from trl import SFTTrainer,SFTConfig
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed
from transformers import EarlyStoppingCallback
import os,random,sys,json,re


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# Random seed for reproducibility
# -----------------------------
def get_truly_random_seed_through_os():
    RAND_SIZE = 4
    random_data = os.urandom(RAND_SIZE)
    return int.from_bytes(random_data, byteorder="big")

seed = get_truly_random_seed_through_os()
set_seed(seed)


# -----------------------------
# Default Unsloth config
# -----------------------------
config = {
    "model": "unsloth/Llama-3.2-1B-Instruct",
    "output_dir": "./llama_9000",
    "delete_output": True,
    "dataset": "TUNE.9000.json",         
    "train_file": "TUNE.9000.json",
    "overwrite": True,
    "lr": 5e-5,
    "optimizer": "paged_adamw_8bit",
    "batch_size": 4,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "scheduler": "linear",
    "max_grad_norm": 1.0,
    "max_length": 2048,
    "dropout_rate": 0.1,
    "gradient_steps": 1,
    "gradient_checkpointing": False,
    "bf16": True,    
    "epochs": 1,
    "logging_steps": 10,
    "patience": 3,
    "stop_threshold": 0.01,
    "seed": -1,
    "use_lora": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "load_4bit": True,
    "load_8bit": False,
    "decoder_layer": "GPT2Block",
    "percent": 100,
    "num_samples": 10000,
    "select_output": "auto",
    "dump": False,
    "overwrite": True,
    "needs_shard": True,
    "attention_type": "flash_attention_2",
    "eval_samples": 1000,
    "dataloader_num_workers": 16,
    "eval_batch_size": 1,
    "save_limit": 2,
    "log_first_step": True,
    "save_strategy": "epoch",
    "log_strategy": "steps",
    "report_to": "none",
    "eval_strategy": "epoch",
    "greater_is_better": False,
}

if(len(sys.argv) > 1):
    with open(sys.argv[1],"r") as jf:
        config = json.load(jf)

for k,v in config.items():
    print(f"{k}: {v}")

if(config["delete_output"] == True):
    print("Deleting output directory")
    os.system(f"rm -rfv {config['output_dir']}")

for k,v in config.items():
    print(f"{k}: {v}")

# -----------------------------
# Dataset formatting
# -----------------------------

def to_text(example, tokenizer, max_length, select_output="output"):
    ex = dict(example)

    if select_output == "auto":
        for cand in ["output", "response", "answer", "assistant"]:
            if cand in ex:
                select_output = cand
                break

    input_data = {}
    if "instruction" in ex and ex["instruction"] is not None:
        input_data["instruction"] = ex["instruction"]
    if "input" in ex and ex["input"] is not None:
        input_data["input"] = ex["input"]

    input_str = "\n".join(
        [f"{k}: {','.join(v) if isinstance(v, list) else v}" for k, v in input_data.items()]
    )
    input_str = f"{input_str.strip()}"
    half_length = config["max_length"]//2
    if(len(input_str) > half_length): input_str = input_str[0:half_length]
    output_text = ex.get(select_output, "")
    if isinstance(output_text, list):
        output_text = "\n\n".join(output_text)
    output_str = f"{str(output_text).strip()}"

    text = f"### Prompt:\n\n{input_str}\n\n### Response:\n\n{output_str}{tokenizer.eos_token}"
    return {"text": text}

# -----------------------------
# Model creation with LoRA
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_model(use_lora=True):
    
    """
    Create a pre-trained model with optional LoRA fine-tuning.
    """
    # Determine dtype based on config
    dtype = torch.bfloat16 if config.get("bf16", False) else torch.float32
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        config["model"],
        max_seq_length=config["max_length"],
        dtype=torch.bfloat16 if config["bf16"] else torch.float16,
        load_in_4bit=config["load_4bit"],
        load_in_8bit=config["load_8bit"],
        
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config["bias"],        
    )
    if config["gradient_checkpointing"] == True:
        print("Gradient Checkpointing: Enabled")
        model.config.use_cache = False
        model.gradient_checkpointing_enable()


    return model, tokenizer


model, tokenizer = create_model()


# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("json", data_files=config["dataset"], split="train").shuffle(seed)

if config["num_samples"] == 0:
    percent = min(max(config["percent"], 1), 100)
    NUM_SAMPLES = int(len(dataset) * (percent / 100))
else:
    NUM_SAMPLES = config["num_samples"]

train_dataset = dataset.select(range(0, NUM_SAMPLES))

if NUM_SAMPLES < len(dataset):
    eval_dataset = dataset.select(range(NUM_SAMPLES, len(dataset)))
    eval_dataset = eval_dataset.select(range(min(len(eval_dataset),int(config["eval_samples"]))))
else:
    # fallback small slice so Trainer doesn’t crash
    eval_dataset = dataset.select(range(0, min(100, len(dataset))))


# Apply mapping once
train_dataset = train_dataset.map(
    lambda ex: to_text(
        ex,
        tokenizer,
        max_length=config["max_length"],
        select_output=config["select_output"]
    ),
    remove_columns=dataset.column_names
)


# Apply mapping once
eval_dataset = eval_dataset.map(
    lambda ex: to_text(
        ex,
        tokenizer,
        max_length=config["max_length"],
        select_output=config["select_output"]
    ),
    remove_columns=dataset.column_names
)


last_checkpoint = None
if os.path.isdir(config["output_dir"]):
    last_checkpoint = get_last_checkpoint(config["output_dir"])
resume = last_checkpoint is not None
if resume:
    print(f"Resuming Checkpoint: {resume}")

N = 2
updates_per_epoch = len(train_dataset) // (config["batch_size"] * config["gradient_steps"])
total_steps = updates_per_epoch * config["epochs"]
eval_steps = updates_per_epoch // N
save_steps = total_steps // 10

print(f"Eval Steps: {eval_steps}")
print(f"Save Steps: {save_steps}")

training_args = SFTConfig(
    output_dir=config["output_dir"],
    overwrite_output_dir=config["overwrite"],
    save_strategy=config["save_strategy"],
    logging_strategy=config["log_strategy"],
    logging_steps=config["logging_steps"],
    learning_rate=config["lr"],
    logging_first_step=config["log_first_step"],
    dataloader_num_workers=config["dataloader_num_workers"],
    optim=config["optimizer"],
    lr_scheduler_type=config["scheduler"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["eval_batch_size"],
    num_train_epochs=config["epochs"],
    gradient_accumulation_steps=config["gradient_steps"],
    bf16=config["bf16"],
    fp16=not config["bf16"],
    max_grad_norm=config["max_grad_norm"],
    report_to=config["report_to"],
    warmup_steps=config["warmup_steps"],
    save_total_limit=config["save_limit"],    
    eval_strategy= config["eval_strategy"],
    metric_for_best_model="eval_loss",    
    greater_is_better=config["greater_is_better"],
    dataset_text_field="text",
    max_seq_length=config["max_length"],        
    auto_find_batch_size=True,
    load_best_model_at_end=True
)

trainer = SFTTrainer(
    model = model,
    dataset_text_field = "text",
    packing = False,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    tokenizer = tokenizer,
    args = training_args,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=config["patience"], early_stopping_threshold=config["stop_threshold"])],    
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "### Prompt:\n\n",
    response_part    = "### Response:\n\n"
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

checkpoint = None
if resume == True:
    checkpoint = last_checkpoint

trainer_stats = trainer.train(resume_from_checkpoint=checkpoint)

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

print("Saving Model....")
model.save_pretrained(config["output_dir"])
tokenizer.save_pretrained(config["output_dir"])

```
