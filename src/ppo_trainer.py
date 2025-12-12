from typing import List
import torch

class PPOTrainer:

    def __init__(self, policy, tokenizer):
        self.policy = policy
        self.tokenizer = tokenizer

    def create_chat_batch_from_prompts(self, prompts: List[str]):
        """
        Generates chat formated prompts from simple prompts
        :param prompts: List of prompts
        :type prompts: List[str]
        """
        chat_formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            chat_formatted_prompt = self.tokenizer.apply_chat_template(messages, 
                                                                       add_generation_prompt = True,
                                                                       tokenize = False)
            chat_formatted_prompts.append(chat_formatted_prompt)
        return chat_formatted_prompts

    def rollout(self, inputs, max_new_tokens = 10):
        """
        Performs rollout on input batch
        
        :param inputs: List of chat formatted inputs
        """
        tokenized_inputs = self.tokenizer(inputs, padding = 'longest', padding_side = "left", return_tensors = "pt")
        outputs = self.policy.generate(**tokenized_inputs, max_new_tokens = max_new_tokens)
        return outputs