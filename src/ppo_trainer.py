from typing import List, Any, Optional, Dict
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.distributions import Categorical
import numpy as np
import wandb

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
        log_probs = self.get_completion_log_probs(completions)
        values = self.get_completion_values(completions)
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
            kl_divergence.append(torch.sum(online_log_probs - offline_log_probs, dim = -1))

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
        B, T, _ = log_probs.shape
        total_entropy = 0
        for i in range(B):
            for t in range(T):
                if mask[i, t] != 0:
                    distr = Categorical(log_probs[i, t, :])
                    total_entropy += distr.entropy()
        total_entropy /= mask.sum()
        return total_entropy
    
    
    def train(self, iterations: int, dataset: Dataset, batch_sampling_percentage: float,
              mini_batch_size: int, epochs: int, max_new_tokens: int, prompt_col_name: str,
              gamma: float, lmbda: float, epsilon: float, value_loss_coef: float, entropy_loss_coef: float,
              wandb_project: Optional[str] = "rlhf-training", wandb_run_name: Optional[str] = None,
              frequency_of_completion_logging: Optional[int] = None):
        
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
        :param value_loss_coef: Coefficient for value loss in totall loss function
        :type value_loss_coef: float
        :param entropy_loss_coef: Coefficient for value loss in totall loss function
        :type entropy_loss_coef: float
        """

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

            # calculate rewards, log_probs and values (on frozen policy)
            mini_batches_log_probs, mini_batches_rewards, mini_batches_values = [], [], []
            for i, (mini_batch_chat_formatted, mini_batch_completions, mini_batch_output_masks) in enumerate(zip(mini_batches_chat_formatted,
                                                                                                   mini_batches_completions,
                                                                                                   mini_batches_output_masks)):
                (log_probs, rewards, values) = self.get_log_probs_rewards_values(mini_batch_chat_formatted, mini_batch_completions)
                mini_batches_log_probs.append(log_probs)
                mini_batches_rewards.append(rewards)
                mini_batches_values.append(values * mini_batches_output_masks[i])
                iter_metrics["rewards"].extend(rewards.flatten().cpu().tolist())

            wandb.log({
                "rewards/mean": np.mean(iter_metrics["rewards"]),
                "rewards/std": np.std(iter_metrics["rewards"]),
                "rewards/min": np.min(iter_metrics["rewards"]),
                "rewards/max": np.max(iter_metrics["rewards"]),
            }, step=global_step)
            self._unfreeze_policy()

            

            for epoch in range(epochs):

                epoch_metrics = {
                    "policy_loss": [],
                    "value_loss": [],
                    "entropy_loss": [],
                    "total_loss": [],
                    "kl_div": [],
                    "clip_frac": [],
                    "approx_kl": [],
                }

                zip_for_mini_batches = zip(mini_batches_chat_formatted, mini_batches_completions, mini_batches_log_probs,
                                           mini_batches_rewards, mini_batches_values, mini_batches_output_masks)

                for (mini_batch_chat_formatted, mini_batch_completions, mini_batch_log_probs, mini_batch_rewards, mini_batch_values, mini_batch_output_masks) in zip_for_mini_batches:

                    # calculate log_probs for unfrozen policy
                    online_policy_log_probs = self.get_completion_log_probs(mini_batch_completions)
                    online_policy_target_log_probs = torch.gather(online_policy_log_probs, dim = -1, index = mini_batch_completions.unsqueeze(-1)).squeeze(-1)
                    offline_policy_target_log_probs = torch.gather(mini_batch_log_probs, dim = -1, index = mini_batch_completions.unsqueeze(-1)).squeeze(-1)

                    # calculate kl divergence
                    kl_divergence = self.calculate_kl_divergence(online_policy_log_probs, mini_batch_log_probs)
                    # update rewards, subtracting kl divergence from rewards
                    rewards = self.get_completion_only_rewards(mini_batch_output_masks, mini_batch_rewards)
                    rewards -= kl_divergence

                    online_values = self.get_completion_values(mini_batch_completions) * mini_batch_output_masks

                    # calculate the advantage for every step, using gae
                    gae = self.calculate_GAE(rewards, mini_batch_values, gamma, lmbda, mini_batch_output_masks)
                    likelihood_ratio = torch.exp(online_policy_target_log_probs - offline_policy_target_log_probs)
                    clipped_likelihood_ratio = torch.clamp(likelihood_ratio, 1 - epsilon, 1 + epsilon)
                    loss = torch.min(likelihood_ratio, clipped_likelihood_ratio) * gae
                    
                    loss = loss.sum() / mini_batch_output_masks.sum()

                    # calculate entropy loss
                    entropy_loss = self.calculate_entropy(online_policy_log_probs, mini_batch_output_masks)

                    # calculate value loss
                    value_loss = 0.5 * (((online_values - (mini_batch_values + gae)) ** 2).mean())

                    total_loss = -loss + value_loss_coef * value_loss - entropy_loss_coef * entropy_loss
                    print(f"Iteration {iter + 1} / {iterations} Epoch {epoch + 1} / {epochs} Total loss: {total_loss.item()}")

                    optimizer.zero_grad()

                    total_loss.backward()

                    optimizer.step()

                    with torch.no_grad():
                        clipping_fraction = ((likelihood_ratio - 1.0).abs() > epsilon).float().mean().item()
                        approx_kl = (offline_policy_target_log_probs - online_policy_target_log_probs).mean().item()

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
                        "train/return_mean": (mini_batch_values + gae).mean().item(),
                        "train/return_std": (mini_batch_values + gae).std().item(),
                        "global_step": global_step,
                        "epoch": epoch,
                    }, step=global_step)
                
                    global_step += 1

                    if (frequency_of_completion_logging is not None) and (iter % frequency_of_completion_logging == 0):
                        sample_prompts = mini_batches[0][:3]
                        sample_completions = mini_batches_completions[0][:3]
                        
                        completion_table = wandb.Table(
                            columns=["iteration", "prompt", "completion"],
                            data=[
                                [iter, prompt, self.tokenizer.decode(completion)]
                                for prompt, completion in zip(sample_prompts, sample_completions)
                            ]
                        )
                        wandb.log({f"samples/completions_iter_{iter}": completion_table}, step=global_step)
        
        wandb.finish()
        print("Training completed!")

        