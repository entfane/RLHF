from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from reward_model import RewardModel

class PPOTrainer:

    def __init__(self, model_name, reward_model_name):
        self.policy = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reward_model = RewardModel(reward_model_name)





