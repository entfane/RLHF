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

    def get_completion_only_values(self, input, output):
        """
        Returns a zeroed out tensor of values only for the completion tokens, having 0 for all the input and padding tokens
        
        :param input: batch of chat formatted inputs
        :param output: batch of full completions - including padding, prompt and completions
        """
        _, _, values = self.policy(output)
        values = self._zero_out_input(input, output, values)
        return values
    
    def get_completion_only_logits(self, input, output):
        """
        Returns a zeroed out tensor of logits only for the completion tokens, having 0 for all the input and padding tokens
        
        :param input: batch of chat formatted inputs
        :param output: batch of full completions - including padding, prompt and completions
        """
        logits, _, _ = self.policy(output)
        logits = self._zero_out_input(input, output, logits)
        return logits
    
    def _zero_out_input(self, input, completion, output):
        """
        Zeroes out input part and padding of the full output tensor, which includes padding, prompt and completion
        
        :param input: batch of chat formatted inputs
        :param output: either values or logits of the full generation (padding, prompt and completion)
        """
        padded_input = self.tokenizer(input, padding = 'longest', padding_side = "left", return_tensors = "pt")["input_ids"]
        _, T = padded_input.shape
        output[:, :T] = 0
        keep_mask = (completion != self.tokenizer.eos_token_id) & (completion != self.tokenizer.pad_token_id)
        zero_tensor = torch.zeros_like(completion)
        output = torch.where(keep_mask, output, zero_tensor)
        return output
    
    def _get_last_token_idx(self, completion):
        """
        Returns a Tensor of last token index.
        
        :param completion: A batch of completions only, with zeroed out paddings, prompt and eos tokens
        """
        mask = (completion != torch.zeros_like(completion))
        _, T = completion.shape
        idxs = torch.arange(T)
        masked_idx = mask * (idxs)
        last_idx = masked_idx.argmax(dim=1)
        return last_idx

