from typing import List
import torch
import torch.nn.functional as F

class PPOTrainer:

    def __init__(self, policy, tokenizer, reward_model):
        self.policy = policy
        self.tokenizer = tokenizer
        self.reward_model = reward_model

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

        log_probs = F.log_softmax(logits, dim = -1)
        target_log_probs = torch.gather(log_probs, dim = -1, index = output.unsqueeze(-1)).squeeze(-1)

        logits = self._zero_out_input(input, output, target_log_probs)
        return logits

    def get_completion_decoded(self, input, completion):
        """
        Returns decoded completions only
        
        :param input: chat formatted and tokenized input batch
        :param completion: chat formatted completion batch, includes both prompt and completion
        """
        _, T = input.shape
        decoded_completions = self.tokenizer.batch_decode(completion[:, T:], skip_special_tokens = True)
        return decoded_completions
    
    def get_completion_only_rewards(self, input, output, rewards):
        """
        Returns a tensor of rewards for completion only
        
        :param input: batch of chat formatted inputs
        :param output: tokenized output consisting of prompt, completion and paddings
        :param rewards: tensor of rewards per every completion
        """
        reward_output = torch.ones_like(output, dtype = torch.float)
        reward_output = self._zero_out_input(input, output, reward_output)
        last_idx = self._get_last_token_idx(reward_output)
        B, _ = output.shape
        for i in range(B):
            reward_output[i, last_idx[i]] = rewards[i]
        return reward_output
        
    
    
    def _zero_out_input(self, input, completion, output):
        """
        Zeroes out input part and padding of the full output tensor, which includes padding, prompt and completion
        
        :param input: batch of chat formatted inputs
        :param completion: full completion including prompt, completion and all the paddings
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

    def get_logits_rewards_values(self, batch, completions):
        tokenized_inputs = self.tokenizer(batch, padding = 'longest', padding_side = "left", return_tensors = "pt")["input_ids"]
        _, T = tokenized_inputs.shape
        completions_only = self.tokenizer.batch_decode(completions[:, T:], skip_special_tokens = True)
        rewards = self.reward_model.get_reward(zip(batch, completions_only))
        logits = self.get_completion_only_logits(batch, completions)
        values = self.get_completion_only_values(batch, completions)
        return rewards, logits, values
        


