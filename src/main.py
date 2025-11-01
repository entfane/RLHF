from policy import Policy
from ppo_trainer import PPOTrainer
from datasets import load_dataset
from reward_model import RewardModel
import torch

def get_completions_only(trajectories, prompts, tokenizer):
    completions = []
    for (prompt, trajectory) in zip(prompts, trajectories):
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                                                                    tokenize = False, add_generation_prompt = True)
        completion = trajectory.replace(prompt, "")
        completions.append(completion)
    return completions
    

if __name__ == "__main__":
    offline_policy = Policy("HuggingFaceTB/SmolLM2-135M-Instruct")
    online_policy = Policy("HuggingFaceTB/SmolLM2-135M-Instruct")
    reward_model = RewardModel("Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    offline_policy.freeze_params()
    trainer = PPOTrainer(offline_policy)
    prompts = load_dataset("data-is-better-together/10k_prompts_ranked", split = "train[:2]")
    prompts = prompts['prompt']
    trajectories = trainer.generate_trajectories(prompts)
    offline_logits, _ = offline_policy.generate_logits_and_values(trajectories)
    online_logits, values = online_policy.generate_logits_and_values(trajectories)
    completions_only = get_completions_only(trajectories, prompts, offline_policy.tokenizer)
    tokenized_completions = offline_policy.tokenizer(completions_only, return_tensors = "pt", padding = True)
    print(tokenized_completions)

