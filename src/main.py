from ppo_trainer import PPOTrainer
from datasets import load_dataset
from reward_model import RewardModel
import torch
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, HfArgumentParser
from datasets import load_dataset
from dataclasses import dataclass, field

@dataclass
class RLHFArguments:
    model_name: str = field(default = "none", metadata={"help": "HF Name of the model to optimize"})
    reward_model_name: str = field(default = "none", metadata={"help": "HF Reward model name"})
    prompt_column_name: str = field(default = "text", metadata={"help": "Name of prompt column in the dataset"})
    dataset: str = field(default = "none", metadata={"help": "HF dataset name"})
    iterations: int = field(default = 1, metadata={"help": "Number of iterations"})
    batch_sampling_percentage: float = field(default = 1, metadata={"help": "Percentage of samples to be taken in a batch in a single iteration"})
    mini_batch_size: int = field(default = 1, metadata={"help": "Mini batch size"})
    epochs: int = field(default = 1, metadata={"help": "Number of epochs"})
    max_new_tokens: int = field(default = 128, metadata={"help": "Maximum new tokens to be generated in rollouts"})

if __name__ == "__main__":
    policy = AutoModelForCausalLMWithValueHead.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    reward_model = RewardModel("Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    dataset = load_dataset("TuringEnterprises/Turing-Open-Reasoning", split = "train")
    trainer = PPOTrainer(policy, tokenizer, reward_model)
    trainer.train(iterations=1, dataset = dataset, batch_sampling_percentage=0.1, mini_batch_size=4,
                  epochs = 1, max_new_tokens = 16, prompt_col_name="question", gamma = 0.1, lmbda = 0.1, epsilon=0.1)