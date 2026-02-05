from typing import List, Any, Optional, Dict
import torch
import torch.nn.functional as F
from datasets import Dataset
import numpy as np
from copy import deepcopy
import wandb

class PPOTrainer:

    def __init__(self, policy, tokenizer, reward_model):
        self.policy = policy
        self.tokenizer = tokenizer
        self.reward_model = reward_model

    def create_chat_batch_from_prompts(self, prompts: List[str], max_input_length: int=128) -> List[str]:
        """
        Generates a list of chat formatted inputs

        :param prompts: List of prompts
        :type prompts: List[str]
        :return: List of chat formatted inputs
        :rtype: List[str]
        """
        chat_formatted_prompts = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, truncation=True, max_length=max_input_length)
            prompt = self.tokenizer.decode(tokens)
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

    def get_completion_values(self, model, completion: torch.Tensor) -> torch.Tensor:
        """
        Generates values for the completions
        
        :param completion: A tensor of full completions consisting of padding, input and output
        :type completion: torch.Tensor
        :return: A tensor of values for each state
        :rtype: Tensor
        """
        _, _, values = model(completion)
        return values.to("cuda")
    
    def get_completion_log_probs(self, model, completion: torch.Tensor) -> torch.Tensor:
        """
        Generates lob probabilities for the completion
        
        :param completion: A tensor of full completions consisting of padding, input and output
        :type completion: torch.Tensor
        :return: A tensor of log probabilities for each state
        :rtype: Tensor
        """
        logits, _, _ = model(completion)
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
    
    def get_completion_only_rewards(self, mask: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """
        For every completion sets the last completion token to the reward value as a terminal state, other states all set to 0

        :param mask: Mask for generated completions only, with 0s for padding/eos tokens
        :type mask: torch.Tensor
        :param rewards: Tensor of rewards for every completion
        :type rewards: torch.Tensor
        """
        last_idx = self._get_last_token_idx_from_mask(mask)
        reward_output = torch.zeros_like(mask, dtype = torch.float)
        B, _ = mask.shape
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
        :param output: Tensor of either log_probs or values for the whole completion, including padding, input and output
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
    
    def _get_last_token_idx_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor of indices of last tokens for every completion. 
        
        :param mask: Mask for generated completions only, with 0s for padding/eos tokens
        :type mask: torch.Tensor
        :return: Tensor of last indices
        :rtype: Tensor
        """
        _, T = mask.shape
        idxs = torch.arange(T).to("cuda")
        masked_idx = mask * (idxs)
        last_idx = masked_idx.argmax(dim=1)
        return last_idx

    def get_log_probs_rewards_values(self, batch: List[str], completions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates log probabilities, rewards for the completion and values for states
        
        :param batch: Batch of chat formatted inputs
        :type batch: List[str]
        :param completions: Tensor of full completions consisting of padding, input and output
        :type completions: torch.Tensor
        :return: Tuple of log probabilities, rewards and values for each state
        :rtype: tuple[Tensor, Tensor, Tensor]
        """
        tokenized_inputs = self.tokenizer(batch, padding = 'longest', padding_side = "left", return_tensors = "pt")["input_ids"]
        _, T = tokenized_inputs.shape
        completions_only = self.tokenizer.batch_decode(completions[:, T:], skip_special_tokens = True)
        rewards = self.reward_model.get_reward(zip(batch, completions_only))
        log_probs = self.get_completion_log_probs(self.policy, completions)
        values = self.get_completion_values(self.policy, completions)
        return (log_probs, rewards, values)
    
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
    
    def calculate_kl_divergence(self, online_policy_log_probs: List[torch.Tensor], offline_policy_log_probs: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculates the KL divergence between the online and offline policies
        
        :param online_policy_log_probs: List of tensors of online policy log probabilities for each token
        :type online_policy_log_probs: List[torch.Tensor]
        :param offline_policy_log_probs: List of tensors of offline policy log probabilities for each token
        :type offline_policy_log_probs: List[torch.Tensor]
        :return: Tensor of KL divergences for each token of each completion
        :rtype: torch.Tensor
        """
        kl_divergence = []
        for (online_log_probs, offline_log_probs) in zip(online_policy_log_probs, offline_policy_log_probs):
            kl_divergence.append(torch.sum(torch.exp(online_log_probs) * (online_log_probs - offline_log_probs), dim = -1))

        return torch.stack(kl_divergence, dim=0)
    
    def calculate_GAE(self, rewards: torch.Tensor, values: torch.Tensor, gamma: float, lmbda: float, mask: torch.Tensor) -> torch.Tensor:
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
        :param mask: Mask for generated completions only, with 0s for padding/eos tokens
        :type mask: torch.Tensor
        :return: GAE tensor
        :rtype: torch.Tensor
        """
        B, _ = rewards.shape
        last_idx = self._get_last_token_idx_from_mask(mask)
        advantage = torch.zeros_like(rewards, dtype=torch.float)
        for i in range(B):
            beta = rewards[i, last_idx[i]] - values[i, last_idx[i]]
            advantage[i, last_idx[i]] = beta
            for j in range(last_idx[i] - 1, 0, -1):
                beta = rewards[i, j] + gamma * values[i, j + 1] - values[i, j]
                advantage[i, j] = beta + lmbda * gamma * advantage[i, j + 1]
        return advantage
    

    def calculate_entropy(self, log_probs: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Calculates entropy for log probabilities
        
        :param log_probs: Log probabilities of tokens
        :type log_probs: torch.Tensor
        :param mask: Mask of output only tokens
        :type mask: torch.Tensor
        :return: Normalized entropy
        :rtype: float
        """
        probs = torch.exp(log_probs).clamp(min = 1e-10, max=1.0)
        # clamp probs in case of -inf log_prob
        safe_log_probs = log_probs.clamp(min = -20, max = 0)
        
        entropy = - torch.sum(probs * safe_log_probs, dim = -1)
        entropy = entropy * mask
        entropy = entropy.sum() / mask.sum()
        return entropy
    
    
    def train(self, iterations: int, dataset: Dataset, batch_sampling_percentage: float,
              mini_batch_size: int, epochs: int, max_new_tokens: int, prompt_col_name: str,
              beta: float, gamma: float, lmbda: float, epsilon: float, value_loss_coef: float, entropy_loss_coef: float,
              wandb_project: Optional[str] = "rlhf-training", wandb_run_name: Optional[str] = None,
              frequency_of_completion_logging: Optional[int] = None, log_wandb: Optional[bool] = False,
              max_input_length: Optional[int] = None):
        
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
        :param beta: Coefficient for KL divergence penalty
        :type beta: float
        :param gamma: Gamma parameter in GAE
        :type gamma: float
        :param lmbda: Lambda parameter in GAE
        :type lmbda: float
        :param epsilon: PPO loss clipping epsilon
        :type epsilon: float
        :param value_loss_coef: Coefficient for value loss in totall loss function
        :type value_loss_coef: float
        :param entropy_loss_coef: Coefficient for value loss in totall loss function
        :type entropy_loss_coef: float
        """
        if log_wandb:
            config = {
                "iterations": iterations,
                "batch_sampling_percentage": batch_sampling_percentage,
                "mini_batch_size": mini_batch_size,
                "epochs": epochs,
                "max_new_tokens": max_new_tokens,
                "gamma": gamma,
                "lambda": lmbda,
                "epsilon": epsilon,
                "value_loss_coef": value_loss_coef,
                "entropy_loss_coef": entropy_loss_coef,
                "dataset_size": len(dataset),
            }

            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=config
            )

            wandb.watch(self.policy.pretrained_model, log="gradients", log_freq=100, log_graph=False)
            wandb.watch(self.policy.v_head, log="gradients", log_freq=100, log_graph=False)

        optimizer = torch.optim.Adam(self.policy.parameters())

        for iter in range(iterations):

            if log_wandb:
                iter_metrics = {
                    "iteration": iter,
                    "policy_losses": [],
                    "value_losses": [],
                    "entropy_losses": [],
                    "total_losses": [],
                    "kl_divergences": [],
                    "rewards": [],
                    "advantages": [],
                    "likelihood_ratios": [],
                    "clipping_fractions": [],
                    "value_predictions": [],
                    "explained_variance": [],
                }

            global_step = 0

            # sample a batch of minibatches
            batch = self.get_random_batch(dataset, percentage=batch_sampling_percentage)[prompt_col_name]
            mini_batches = self._get_mini_batches(batch, mini_batch_size=mini_batch_size)


            batches_rollouts = []
            batches_output_masks = []
            batches_offline_log_probs = []
            batches_rewards = []
            batches_offline_values = []
            batches_offline_target_log_probs = []
            with torch.no_grad():
                for mini_batch in mini_batches:
                    chat_formatted_mini_batch = self.create_chat_batch_from_prompts(mini_batch, max_input_length = max_input_length)
                    rollouts = self.rollout(chat_formatted_mini_batch, max_new_tokens=max_new_tokens).detach()
                    mini_batch_output_masks = self._get_output_only_mask(chat_formatted_mini_batch, rollouts).detach()
                    (offline_log_probs, rewards, offline_values) = self.get_log_probs_rewards_values(chat_formatted_mini_batch, rollouts)
                    rollouts = rollouts.cpu()
                    mini_batch_output_masks = mini_batch_output_masks.cpu()
                    offline_log_probs = offline_log_probs.cpu()
                    rewards = rewards.cpu()
                    offline_values = offline_values.cpu()
                    offline_target_log_probs = torch.gather(offline_log_probs, dim = -1, index = rollouts.unsqueeze(-1)).squeeze(-1).detach().cpu()

                    batches_rollouts.append(rollouts)
                    batches_output_masks.append(mini_batch_output_masks)
                    batches_offline_log_probs.append(offline_log_probs)
                    batches_rewards.append(rewards)
                    batches_offline_values.append(offline_values)
                    batches_offline_target_log_probs.append(offline_target_log_probs)

                    del rollouts, mini_batch_output_masks, offline_log_probs, rewards, offline_values, offline_target_log_probs
                    torch.cuda.empty_cache()

            for epoch in range(epochs):

                if log_wandb:
                    epoch_metrics = {
                        "policy_loss": [],
                        "value_loss": [],
                        "entropy_loss": [],
                        "total_loss": [],
                        "kl_div": [],
                        "clip_frac": [],
                        "approx_kl": [],
                    }

                for step in range(len(mini_batches)):

                    rollouts = batches_rollouts[step].to('cuda')
                    mini_batch_output_masks = batches_output_masks[step].to('cuda')
                    offline_log_probs = batches_offline_log_probs[step].to('cuda')
                    rewards = batches_rewards[step].to('cuda')
                    offline_values = batches_offline_values[step].to('cuda')
                    offline_target_log_probs = batches_offline_target_log_probs[step].to('cuda')

                    # calculate log_probs for unfrozen policy
                    online_policy_log_probs = self.get_completion_log_probs(self.policy, rollouts)
                    online_policy_target_log_probs = torch.gather(online_policy_log_probs, dim = -1, index = rollouts.unsqueeze(-1)).squeeze(-1)


                    # calculate kl divergence
                    with torch.no_grad():
                        kl_divergence = self.calculate_kl_divergence(online_policy_log_probs, offline_log_probs)

                    # update rewards, subtracting kl divergence from rewards
                    rewards = self.get_completion_only_rewards(mini_batch_output_masks, rewards) 
                    rewards -= beta * kl_divergence
                    
                    online_values = self.get_completion_values(self.policy, rollouts) * mini_batch_output_masks

                    # calculate the advantage for every step, using gae
                    gae = self.calculate_GAE(rewards, offline_values, gamma, lmbda, mini_batch_output_masks).detach()
                    likelihood_ratio = torch.exp(online_policy_target_log_probs - offline_target_log_probs)
                    clipped_likelihood_ratio = torch.clamp(likelihood_ratio, 1 - epsilon, 1 + epsilon)
                    loss = torch.min(likelihood_ratio, clipped_likelihood_ratio) * gae
                    
                    loss = loss.sum() / mini_batch_output_masks.sum()

                    # calculate entropy loss
                    entropy_loss = self.calculate_entropy(online_policy_log_probs, mini_batch_output_masks)

                    # calculate value loss
                    returns = (offline_values + gae).detach()
                    value_loss = 0.5 * (((online_values - returns) ** 2).mean())
                    print(f"Debug step {step}:")
                    print(f"  loss: {loss.item()}")
                    print(f"  value_loss: {value_loss.item()}")
                    print(f"  entropy_loss: {entropy_loss.item()}")
                    print(f"  rewards min/max: {rewards.min().item()}/{rewards.max().item()}")
                    print(f"  gae min/max: {gae.min().item()}/{gae.max().item()}")
                    print(f"  likelihood_ratio min/max: {clipped_likelihood_ratio.min().item()}/{clipped_likelihood_ratio.max().item()}")
                    print(f"  kl_divergence: {kl_divergence.mean().item()}")

                    total_loss = -loss + value_loss_coef * value_loss - entropy_loss_coef * entropy_loss
                    print(f"Iteration {iter + 1} / {iterations} Epoch {epoch + 1} / {epochs} Step {step + 1} / {len(mini_batches)} Total loss: {total_loss.item()}")

                    optimizer.zero_grad()

                    total_loss.backward()

                    optimizer.step()
                    if log_wandb:
                        with torch.no_grad():
                            clipping_fraction = ((likelihood_ratio - 1.0).abs() > epsilon).float().mean().item()
                            approx_kl = (offline_target_log_probs - online_policy_target_log_probs).mean().item()

                        epoch_metrics["policy_loss"].append(loss.item())
                        epoch_metrics["value_loss"].append(value_loss.item())
                        epoch_metrics["entropy_loss"].append(entropy_loss.item())
                        epoch_metrics["total_loss"].append(total_loss.item())
                        epoch_metrics["kl_div"].append(kl_divergence.mean().item())
                        epoch_metrics["approx_kl"].append(approx_kl)
                        epoch_metrics["clip_frac"].append(clipping_fraction)

                        wandb.log({
                            "train/policy_loss": loss.item(),
                            "train/value_loss": value_loss.item(),
                            "train/entropy_loss": entropy_loss.item(),
                            "train/total_loss": total_loss.item(),
                            "train/kl_divergence": kl_divergence.mean().item(),
                            "train/clipping_fraction": clipping_fraction,
                            "train/approx_kl": approx_kl,
                            "train/likelihood_ratio_mean": likelihood_ratio.mean().item(),
                            "train/advantage_mean": gae.mean().item(),
                            "train/advantage_std": gae.std().item(),
                            "train/learning_rate": optimizer.param_groups[0]['lr'],
                            "train/value_pred_mean": online_values.mean().item(),
                            "train/value_pred_std": online_values.std().item(),
                            "train/return_mean": (offline_values + gae).mean().item(),
                            "train/return_std": (offline_values + gae).std().item(),
                            "global_step": global_step,
                            "epoch": epoch,
                        }, step=global_step)
                    
                        global_step += 1

                        if (frequency_of_completion_logging is not None) and (iter % frequency_of_completion_logging == 0):
                            sample_prompts = mini_batches[0][:3]
                            sample_completions = rollouts[0][:3]
                            
                            completion_table = wandb.Table(
                                columns=["iteration", "prompt", "completion"],
                                data=[
                                    [iter, prompt, self.tokenizer.decode(completion)]
                                    for prompt, completion in zip(sample_prompts, sample_completions)
                                ]
                            )
                            wandb.log({f"samples/completions_iter_{iter}": completion_table}, step=global_step)
                    del rollouts, mini_batch_output_masks, offline_log_probs, rewards, offline_values, offline_target_log_probs
                    del online_policy_log_probs, online_policy_target_log_probs, online_values, gae, likelihood_ratio, clipped_likelihood_ratio, kl_divergence, entropy_loss, returns, value_loss, total_loss
                    torch.cuda.empty_cache()
        
        if log_wandb:
            wandb.finish()
        print("Training completed!")

        