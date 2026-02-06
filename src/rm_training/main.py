from rm_trainer import RMTrainer
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RMArguments:
    model_name: str = field(default = "none", metadata={"help": "HF Name of the model"})
    dataset: str = field(default = "none", metadata={"help": "HF dataset name"})
    dataset_split: str = field(default = "train", metadata={"help": "HF dataset split name"})
    prompt_column_name: str = field(default = "prompt", metadata={"help": "Name of prompt column in the dataset"})
    chosen_column_name: str = field(default = "chosen", metadata={"help": "Name of chosen output column in the dataset"})
    rejected_column_name: str = field(default = "rejected", metadata={"help": "Name of rejected output column in the dataset"})
    epochs: int = field(default = 1, metadata={"help": "Number of epochs"})
    mini_batch_size: int = field(default = 1, metadata={"help": "Mini batch size"})
    log_wandb: bool = field(default = False, metadata={"help": "Whether to log in wandb"})
    wandb_project: Optional[str] = field(default = "rm-training", metadata={"help": "Wandb project name"})
    wandb_run_name: Optional[str] = field(default = None, metadata={"help": "Wandb run name"})
    max_len: Optional[int] = field(default = 128, metadata={"help": "Maximum length of input"})
    lr: float = field(default = 1e-6, metadata={"help": "Learning rate"})


if __name__ == "__main__":
    parser = HfArgumentParser(RMArguments)
    args = parser.parse_args_into_dataclasses()[0]
    rm_trainer = RMTrainer(args.model_name)
    rm_trainer.train(dataset_name=args.dataset, split_name=args.dataset_split, prompt_col=args.prompt_column_name,
                     chosen_col=args.chosen_column_name, rejected_col=args.rejected_column_name, epochs=args.epochs, mini_batch_size=args.mini_batch_size,
                     max_len=args.max_len, lr=args.lr, log_wandb=args.log_wandb)
    

    