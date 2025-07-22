from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch

class PPOTrainer:

    def __init__(self, model_name, reward_model_name):
        self.policy = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name, device_map = "auto", num_labels=1)
        self.reward_model_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

    def get_reward(self, prompt, response):
        conv = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
        input = self.reward_model_tokenizer.apply_chat_template(conv, tokenize = False)
        input = self.reward_model_tokenizer(input, return_tensors="pt").to(self.reward_model.device)
        with torch.no_grad():
            score = self.reward_model(**input)
        return score.logits[0][0]
    



