from policy import Policy
from ppo_trainer import PPOTrainer

from datasets import load_dataset

if __name__ == "__main__":
    offline_policy = Policy("Qwen/Qwen3-0.6B")
    online_policy = Policy("Qwen/Qwen3-0.6B")
    offline_policy.freeze_params()
    trainer = PPOTrainer(offline_policy)
    prompts = load_dataset("data-is-better-together/10k_prompts_ranked", split = "train[:2]")
    prompts = prompts['prompt']
    trajectories = trainer.generate_trajectories(prompts)
    offline_logits, _ = offline_policy.generate_logits_and_values(trajectories)
    online_logits, values = online_policy.generate_logits_and_values(trajectories)