from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Tuple

class RewardModel:

    def __init__(self, reward_model_name: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(reward_model_name, device_map = "auto", num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

    @torch.no_grad()
    def get_reward(self, prompt_response_pairs: List[Tuple[str, str]]):
        inputs = []
        for prompt, response in prompt_response_pairs:
            conv = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]
            input = self.tokenizer.apply_chat_template(conv, tokenize = False)
            inputs.append(input)
        inputs = self.tokenizer(inputs, return_tensors="pt", padding = True).to(self.model.device)
        with torch.no_grad():
            score = self.model(**inputs)
        return score.logits