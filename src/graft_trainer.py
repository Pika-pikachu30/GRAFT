import os
import yaml
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class GRAFTTrainerConfig:
    base_model: str = "mistralai/Mistral-7B-v0.1"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    fp16: bool = True
    output_dir: str = "./models/checkpoints"

class GRAFTTrainer:
    """
    Handles Phase 3 of GRAFT: Fine-Tuning Pipeline.
    Fine-tunes an LLM using the graph-structured RAFT dataset.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = GRAFTTrainerConfig()
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                if "model" in yaml_data:
                    self.config.base_model = yaml_data["model"].get("base_model", self.config.base_model)
                    self.config.use_lora = yaml_data["model"].get("use_lora", self.config.use_lora)
                if "training" in yaml_data:
                    for k, v in yaml_data["training"].items():
                        if hasattr(self.config, k):
                            setattr(self.config, k, v)
                if "lora" in yaml_data:
                    for k, v in yaml_data["lora"].items():
                        if hasattr(self.config, f"lora_{k}"):
                            setattr(self.config, f"lora_{k}", v)

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self):
        """Loads base model with QLoRA configuration."""
        logger.info(f"Loading base model {self.config.base_model}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if self.config.fp16 else torch.float32
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            padding_side="right",
            add_eos_token=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.config.use_cache = False

        if self.config.use_lora:
            self.model = prepare_model_for_kbit_training(self.model)
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
        logger.info("Model loaded successfully.")
        return self.model, self.tokenizer

    def setup_trainer(self, dataset_path: str):
        """Sets up the SFTTrainer with custom data collator and dataset formatting."""
        logger.info(f"Loading dataset from {dataset_path}...")
        
        # Load JSONL dataset
        train_ds = load_dataset("json", data_files=os.path.join(dataset_path, "train.jsonl"), split="train")
        val_ds = load_dataset("json", data_files=os.path.join(dataset_path, "val.jsonl"), split="train")

        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example['instruction'])):
                text = f"### Instruction:\n{example['instruction'][i]}\n\n### Input:\n{example['input'][i]}\n\n### Response:\n{example['output'][i]}"
                output_texts.append(text)
            return output_texts

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            num_train_epochs=self.config.num_epochs,
            optim="paged_adamw_32bit",
            save_steps=50,
            save_total_limit=3,
            evaluation_strategy="steps",
            eval_steps=50,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler,
            fp16=self.config.fp16,
            report_to="none" # Switch to 'wandb' or 'tensorboard' for actual tracking
        )

        response_template = "### Response:\n"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            peft_config=None, # Already applied in setup_model
            max_seq_length=self.config.max_seq_length,
            tokenizer=self.tokenizer,
            args=training_args,
            formatting_func=formatting_prompts_func,
            data_collator=collator
        )

        logger.info("Trainer setup complete.")
        return self.trainer

    def train(self):
        """Runs the fine-tuning loop."""
        if not self.trainer:
            raise ValueError("Trainer not set up. Call `setup_trainer()` first.")
        logger.info("Starting training...")
        self.trainer.train()
        logger.info("Training finished. Saving last checkpoint.")
        self.trainer.save_model(os.path.join(self.config.output_dir, "final_checkpoint"))

    def evaluate_checkpoint(self, checkpoint_path: str, eval_dataset_path: str):
        """Evaluates a checkpoint on a held-out set, returning perplexity."""
        # Simple proxy computation for val loss natively via huggingface
        eval_ds = load_dataset("json", data_files=eval_dataset_path, split="train")
        metrics = self.trainer.evaluate(eval_dataset=eval_ds)
        perplexity = torch.exp(torch.tensor(metrics["eval_loss"])).item()
        logger.info(f"Evaluation finished: Loss = {metrics['eval_loss']:.4f}, Perplexity = {perplexity:.2f}")
        return perplexity

    def export_merged_model(self, output_path: str):
        """Merges LoRA weights securely and saves to output directory."""
        logger.info(f"Merging LoRA weights from {self.config.output_dir} and saving to {output_path}...")
        from peft import PeftModel
        
        base_model_loaded = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto"
        )
        peft_model = PeftModel.from_pretrained(base_model_loaded, os.path.join(self.config.output_dir, "final_checkpoint"))
        
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Merged model exported to {output_path}")
