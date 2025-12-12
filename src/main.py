from ppo_trainer import PPOTrainer
from datasets import load_dataset
from reward_model import RewardModel
import torch
from trl import AutoModelForCausalLMWithValueHead
    

if __name__ == "__main__":
    policy = AutoModelForCausalLMWithValueHead.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    reward_model = RewardModel("Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    print(reward_model.get_reward([("Hey, how are you?", "Hey, i am good"), ("What's up?", "All good, dawg!")]))
