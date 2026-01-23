from ppo_trainer import PPOTrainer
from datasets import load_dataset
from reward_model import RewardModel
import torch
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, HfArgumentParser
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RLHFArguments:
    model_name: str = field(default = "none", metadata={"help": "HF Name of the model to optimize"})
    reward_model_name: str = field(default = "none", metadata={"help": "HF Reward model name"})
    dataset: str = field(default = "none", metadata={"help": "HF dataset name"})
    dataset_split: str = field(default = "train", metadata={"help": "HF dataset split name"})
    prompt_column_name: str = field(default = "text", metadata={"help": "Name of prompt column in the dataset"})
    iterations: int = field(default = 1, metadata={"help": "Number of iterations"})
    epochs: int = field(default = 1, metadata={"help": "Number of epochs"})
    batch_sampling_percentage: float = field(default = 1, metadata={"help": "Percentage of samples to be taken in a batch in a single iteration"})
    mini_batch_size: int = field(default = 1, metadata={"help": "Mini batch size"})
    max_new_tokens: int = field(default = 128, metadata={"help": "Maximum new tokens to be generated in rollouts"})
    gamma: float = field(default = 0.98, metadata={"help": "Gamma parameter for TD calculation"})
    lmbda: float = field(default = 0.9, metadata={"help": "Lambda parameter for GAE calculation"})
    epsilon: float = field(default = 0.1, metadata={"help": "Epsilon clipping parameter"})
    value_loss_coef: float = field(default = 0.1, metadata={"help": "Coefficient for value loss in total loss"})
    entropy_loss_coef: float = field(default = 0.1, metadata={"help": "Coefficient for entropy loss in total loss"})
    wandb_project: Optional[str] = field(default = "rlhf-training", metadata={"help": "Wandb project name"})
    wandb_run_name: Optional[str] = field(default = None, metadata={"help": "Wandb run name"})
    frequency_of_completion_logging: Optional[int] = field(default = "None", metadata={"help": "Frequency of completion logging. Measured in iterations"})


if __name__ == "__main__":
    parser = HfArgumentParser(RLHFArguments)
    args = parser.parse_args_into_dataclasses()[0]

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name, device_map="cuda")
    print(f"Model loaded on {policy.pretrained_model.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    reward_model = RewardModel(args.reward_model_name)

    dataset = load_dataset(args.dataset, split = args.dataset_split)
    print("Dataset was loaded")

    trainer = PPOTrainer(policy, tokenizer, reward_model)
    print("PPO Trainer was initialized, starting training...")
    trainer.train(iterations=args.iterations, dataset = dataset, batch_sampling_percentage=args.batch_sampling_percentage, mini_batch_size=args.mini_batch_size,
                  epochs = args.epochs, max_new_tokens = args.max_new_tokens, prompt_col_name=args.prompt_column_name, gamma = args.gamma, lmbda = args.lmbda,
                  epsilon = args.epsilon, value_loss_coef = args.value_loss_coef, entropy_loss_coef = args.entropy_loss_coef, wandb_project = args.wandb_project,
                  wandb_run_name = args.wandb_run_name, frequency_of_completion_logging = args.frequency_of_completion_logging)
    