from ppo_trainer import PPOTrainer
from datasets import load_dataset
from reward_model import RewardModel
import torch
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
    

if __name__ == "__main__":
    policy = AutoModelForCausalLMWithValueHead.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    reward_model = RewardModel("Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    trainer = PPOTrainer(policy, tokenizer)
    inputs = ["how are you?", "How old are you? How is your life going?"]
    batches = trainer.create_chat_batch_from_prompts(inputs)
    outputs = trainer.rollout(batches)