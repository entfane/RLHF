from typing import List
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.distributions import Categorical

class PPOTrainer:

    def __init__(self, policy, tokenizer, reward_model):
        self.policy = policy
        self.tokenizer = tokenizer
        self.reward_model = reward_model

    def create_chat_batch_from_prompts(self, prompts: List[str]) -> List[str]:
        """
        Generates a list of chat formatted inputs

        :param prompts: List of prompts
        :type prompts: List[str]
        :return: List of chat formatted inputs
        :rtype: List[str]
        """
        chat_formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            chat_formatted_prompt = self.tokenizer.apply_chat_template(messages, 
                                                                       add_generation_prompt = True,
                                                                       tokenize = False)
            chat_formatted_prompts.append(chat_formatted_prompt)
        return chat_formatted_prompts

    def rollout(self, inputs: List[str], max_new_tokens: int = 10) -> torch.Tensor:
        """
        Performs a rollout for the batched inputs. Generates completions
        
        :param inputs: Batch of chat formatted inputs
        :type inputs: List[str]
        :param max_new_tokens: Max new tokens to generate
        :type max_new_tokens: int
        :return: A tensor of full completions consisting of padding, input and output
        :rtype: Tensor
        """
        tokenized_inputs = self.tokenizer(inputs, padding = 'longest', padding_side = "left", return_tensors = "pt").to('cuda')
        outputs = self.policy.generate(**tokenized_inputs, max_new_tokens = max_new_tokens)
        return outputs

    def get_completion_values(self, completion: torch.Tensor) -> torch.Tensor:
        """
        Generates values for the completions
        
        :param completion: A tensor of full completions consisting of padding, input and output
        :type completion: torch.Tensor
        :return: A tensor of values for each state
        :rtype: Tensor
        """
        _, _, values = self.policy(completion)
        return values
    
    def get_completion_log_probs(self, completion: torch.Tensor) -> torch.Tensor:
        """
        Generates lob probabilities for the completion
        
        :param completion: A tensor of full completions consisting of padding, input and output
        :type completion: torch.Tensor
        :return: A tensor of log probabilities for each state
        :rtype: Tensor
        """
        logits, _, _ = self.policy(completion)
        log_probs = F.log_softmax(logits, dim = -1)
        return log_probs

    def get_completion_decoded(self, input: torch.Tensor, completion: torch.Tensor) -> List[str]:
        """
        Returns a decoded completion only
        
        :param input: Encoded chat formatted input batch
        :type input: torch.Tensor
        :param completion: Tensor of full completions consisting of padding, input and output
        :type completion: torch.Tensor
        :return: List of completions only
        :rtype: List[str]
        """
        _, T = input.shape
        decoded_completions = self.tokenizer.batch_decode(completion[:, T:], skip_special_tokens = True)
        return decoded_completions
    
    def get_completion_only_rewards(self, input: List[str], output: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """
        Generates rewards only for the last token of the output generation
        
        :param input: Batch of chat formatted inputs 
        :type input: List[str]
        :param output: Tensor of full completions consisting of padding, input and output
        :type output: torch.Tensor
        :param rewards: Tensor of rewards, one per completion
        :type rewards: torch.Tensor
        :return: Tensor of output shape, with rewards at every completion state
        :rtype: Tensor
        """
        reward_output = torch.ones_like(output, dtype = torch.float)
        reward_output = self._zero_out_input(input, output, reward_output)
        last_idx = self._get_last_token_idx(reward_output)
        reward_output = torch.zeros_like(output, dtype = torch.float)
        B, _ = output.shape
        for i in range(B):
            reward_output[i, last_idx[i]] = rewards[i]
        return reward_output
        
    
    
    def _zero_out_input(self, input: List[str], completion: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Zeroes out input part of the output and paddings
        
        :param input: Batch of chat formatted inputs
        :type input: List[str]
        :param completion: Tensor of full completions consisting of padding, input and output
        :type completion: torch.Tensor
        :param output: Tensor of either logits or values for the whole completion, including padding, input and output
        :type output: torch.Tensor
        :return: Zeroed out output
        :rtype: Tensor
        """
        padded_input = self.tokenizer(input, padding = 'longest', padding_side = "left", return_tensors = "pt")["input_ids"]
        _, T = padded_input.shape
        output[:, :T] = 0
        keep_mask = (completion != self.tokenizer.eos_token_id) & (completion != self.tokenizer.pad_token_id)
        zero_tensor = torch.zeros_like(completion)
        output = torch.where(keep_mask, output, zero_tensor)
        return output
    
    def _get_output_only_mask(self, input: List[str], completion: torch.Tensor) -> torch.Tensor:
        """
        Returns a mask for completions only. The mask represents 1's where the token in completion tensor is a rollout completion
        token and 0 where it is input token or padding/eos token.
        
        :param input: Batch of chat formatted inputs
        :type input: List[str]
        :param completion: Tensor of full completions consisting of padding, input and output
        :type completion: torch.Tensor
        :return: Mask for generated completions only, with 0s for padding/eos tokens
        :rtype: Tensor
        """
        output = torch.ones_like(completion)
        padded_input = self.tokenizer(input, padding = 'longest', padding_side = "left", return_tensors = "pt")["input_ids"]
        _, T = padded_input.shape
        output[:, :T] = 0
        keep_mask = (completion != self.tokenizer.eos_token_id) & (completion != self.tokenizer.pad_token_id)
        output = torch.where(keep_mask, output, 0)
        return output
    
    def _get_last_token_idx(self, completion: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor of indices of last tokens for every completion. 
        
        :param completion: Tensor of full completions consisting of padding, input and output
        :type completion: torch.Tensor
        :return: Tensor of last indices
        :rtype: Tensor
        """
        mask = (completion != torch.zeros_like(completion))
        _, T = completion.shape
        idxs = torch.arange(T)
        masked_idx = mask * (idxs)
        last_idx = masked_idx.argmax(dim=1)
        return last_idx
    
    def _freeze_policy(self):
        """
        Freezes the policy, leaves value head trainable
        """
        for param in self.policy.pretrained_model.parameters():
            param.requires_grad = False

        for param in self.policy.v_head.parameters():
            param.requires_grad = True

    def _unfreeze_policy(self):
        """
        Unfreezes the policy
        """
        for param in self.policy.pretrained_model.parameters():
            param.requires_grad = True

    def get_logits_rewards_values(self, batch: List[str], completions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates logits, rewards, values for completions only
        
        :param batch: Batch of chat formatted inputs
        :type batch: List[str]
        :param completions: Tensor of full completions consisting of padding, input and output
        :type completions: torch.Tensor
        :return: Tuple of logits, rewards and values for each state
        :rtype: tuple[Tensor, Tensor, Tensor]
        """
        tokenized_inputs = self.tokenizer(batch, padding = 'longest', padding_side = "left", return_tensors = "pt")["input_ids"]
        _, T = tokenized_inputs.shape
        completions_only = self.tokenizer.batch_decode(completions[:, T:], skip_special_tokens = True)
        rewards = self.reward_model.get_reward(zip(batch, completions_only))
        logits = self.get_completion_log_probs(completions)
        values = self.get_completion_values(completions)
        return (logits, rewards, values)
    
    def get_random_batch(self, dataset: Dataset, percentage: float) -> dict:
        """
        Returns a fraction of the datasets randomly shuffled
        
        :param dataset: Dataset
        :type dataset: Dataset
        :param percentage: Fraction of dataset to be taken
        :type percentage: float
        :return: Dictionary of the dataset
        :rtype: dict
        """
        return dataset.shuffle().select(range(int(percentage * len(dataset))))
    
    def _get_mini_batches(self, batch: List[str], mini_batch_size: int) -> List[List[str]]:
        """
        Creates a list of mini batches from a big batch
        
        :param batch: Initial batch
        :type batch: List[str]
        :param mini_batch_size: Size of mini batches
        :type mini_batch_size: int
        :return: Batch of mini batches
        :rtype: List[List[str]]
        """
        output = []
        for i in range(0, len(batch), mini_batch_size):
            mini_batch = batch[i : (i + mini_batch_size)]
            output.append(mini_batch)
        return output
    
    def calculate_kl_divergence(self, online_policy_logits: List[torch.Tensor], offline_policy_logits: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Calculates KL divergence for all the completion tokens between offline policy and online policy.
        
        :param online_policy_logits: Logits of online policy
        :type online_policy_logits: List[torch.Tensor]
        :param offline_policy_logits: Logits of frozen policy
        :type offline_policy_logits: List[torch.Tensor]
        :return: KL divergence
        :rtype: List[Tensor]
        """
        kl_divergence = []
        for (online_logits, offline_logits) in zip(online_policy_logits, offline_policy_logits):
            kl_divergence.append(online_logits - offline_logits)

        return torch.stack(kl_divergence, dim=0)
    
    def calculate_GAE(self, rewards: torch.Tensor, values: torch.Tensor, gamma: float, lmbda: float) -> torch.Tensor:
        """
        Calculates GAE iteratively from the last state backwards.
        
        :param rewards: Tensor of rewards
        :type rewards: torch.Tensor
        :param values: Tensor of values
        :type values: torch.Tensor
        :param gamma: Gamma parameter
        :type gamma: float
        :param lmbda: Lambda parameter
        :type lmbda: float
        :return: GAE tensor
        :rtype: Tensor
        """
        B, _ = rewards.shape
        last_idx = self._get_last_token_idx(rewards)
        advantage = torch.zeros_like(rewards, dtype=torch.float)
        for i in range(B):
            beta = rewards[i, last_idx[i]] - values[i, last_idx[i]]
            advantage[i, last_idx[i]] = beta
            for j in range(last_idx[i] - 1, 0, -1):
                beta = rewards[i, j] + gamma * values[i, j + 1] - values[i, j]
                advantage[i, j] = beta + lmbda * gamma * advantage[i, j + 1]
        return advantage
    

    def calculate_entropy(self, input: List[str], output: torch.Tensor) -> float:
        """
        Calculates entropy loss for completions only.
        
        :param input: Batch of chat formatted inputs
        :type input: List[str]
        :param output: Tensor of full completions consisting of padding, input and output
        :type output: torch.Tensor
        :return: Entropy
        :rtype: float
        """

        logits, _, _ = self.policy(output)
        mask = torch.ones_like(output)
        mask = self._zero_out_input(input, output, mask)
        B, T, C = logits.shape
        for i in range(B):
            total_entropy_sum = 0
            total_states = 0
            for t in range(T):
                if mask[i, t] != 0:
                    total_states += 1
                    distr = Categorical(F.log_softmax(logits[i, t, :]))
                    total_entropy_sum -= distr.entropy()
            total_entropy_sum /= total_states
        total_entropy_sum /= B
        return total_entropy_sum


    
    
    def train(self, iterations: int, dataset: Dataset, batch_sampling_percentage: float,
              mini_batch_size: int, epochs: int, max_new_tokens: int, prompt_col_name: str,
              gamma: float, lmbda: float, epsilon: float):
        
        """
        Main RLHF training loop
        
        :param iterations: Number of training iterations
        :type iterations: int
        :param dataset: Training Dataset
        :type dataset: Dataset
        :param batch_sampling_percentage: Percentage of dataset to be used for training in 1 iteration
        :type batch_sampling_percentage: float
        :param mini_batch_size: Mini batch size
        :type mini_batch_size: int
        :param epochs: Number of epochs per iteration
        :type epochs: int
        :param max_new_tokens: Max new tokens to be generated in rollout
        :type max_new_tokens: int
        :param prompt_col_name: Name of the column with prompts
        :type prompt_col_name: str
        :param gamma: Gamma parameter in GAE
        :type gamma: float
        :param lmbda: Lambda parameter in GAE
        :type lmbda: float
        :param epsilon: PPO loss clipping epsilon
        :type epsilon: float
        """

        for iter in range(iterations):

            # sample a batch of minibatches
            self._freeze_policy()
            batch = self.get_random_batch(dataset, percentage=batch_sampling_percentage)[prompt_col_name]
            mini_batches = self._get_mini_batches(batch, mini_batch_size=mini_batch_size)

            # perform a rollout on batches
            mini_batches_chat_formatted = []
            mini_batches_completions = []
            mini_batches_output_masks = []
            for mini_batch in mini_batches:
                chat_formatted_mini_batch = self.create_chat_batch_from_prompts(mini_batch)
                mini_batches_chat_formatted.append(chat_formatted_mini_batch)
                rollouts = self.rollout(chat_formatted_mini_batch, max_new_tokens=max_new_tokens)
                mini_batches_completions.append(rollouts)
                mini_batches_output_masks.append(self._get_output_only_mask(chat_formatted_mini_batch, rollouts))
                break

            # calculate rewards, logits and values (on frozen policy)
            mini_batches_logits, mini_batches_rewards, mini_batches_values = [], [], []
            for (mini_batch_chat_formatted, mini_batch_completions, mini_batch_output_masks) in zip(mini_batches_chat_formatted,
                                                                                                   mini_batches_completions,
                                                                                                   mini_batches_output_masks):
                (logits, rewards, values) = self.get_logits_rewards_values(mini_batch_chat_formatted, mini_batch_completions)
                mini_batches_logits.append(logits)
                mini_batches_rewards.append(rewards)
                mini_batches_values.append(values * mini_batch_output_masks)

            self._unfreeze_policy()

            optimizer = torch.optim.Adam(self.policy.parameters())

            for epoch in range(epochs):

                zip_for_mini_batches = zip(mini_batches_chat_formatted, mini_batches_completions, mini_batches_logits,
                                           mini_batches_rewards, mini_batches_values, mini_batches_output_masks)

                for (mini_batch_chat_formatted, mini_batch_completions, mini_batch_logits, mini_batch_rewards, mini_batch_values, mini_batch_output_masks) in zip_for_mini_batches:

                    # calculate logits for unfrozen policy
                    online_policy_logits = self.get_completion_only_log_probs(mini_batch_chat_formatted, mini_batch_completions)


                    # calculate kl divergence
                    kl_divergence = self.calculate_kl_divergence(online_policy_logits, mini_batch_logits)

                    # update rewards, subtracting kl divergence from rewards
                    rewards = self.get_completion_only_rewards(mini_batch_chat_formatted, mini_batch_completions, mini_batch_rewards)
                    rewards -= kl_divergence

                    values = self.get_completion_only_values(mini_batch_chat_formatted, mini_batch_completions)

                    # calculate the advantage for every step, using gae
                    gae = self.calculate_GAE(rewards, mini_batch_values, gamma, lmbda)

                    gae = self._zero_out_input(mini_batch_chat_formatted, mini_batch_completions, gae)

                    # calculate clipped loss
                    clipped_loss = torch.clamp(torch.exp(online_policy_logits - mini_batch_logits), 1 - epsilon, 1 + epsilon) * gae
                    loss =  torch.exp(online_policy_logits - mini_batch_logits) * gae
                    loss = torch.min(loss, clipped_loss)

                    mask = (loss != 0).float()
                    loss = -(loss.sum()) / mask.sum()

                    # calculate entropy loss
                    entropy_loss = self.calculate_entropy(mini_batch_chat_formatted, mini_batch_completions)

                    # calculate value loss
                    value_loss = 0.5 * (((values - (mini_batch_values + gae)) ** 2).mean())

                    total_loss = loss + value_loss - entropy_loss

                    optimizer.zero_grad()

                    total_loss.backward()

                    optimizer.step()