from policy import OfflinePolicy
from ppo_trainer import PPOTrainer

from datasets import load_dataset

if __name__ == "__main__":
    offline_policy = OfflinePolicy("Qwen/Qwen3-0.6B")
    trainer = PPOTrainer(offline_policy)
    prompts = load_dataset("data-is-better-together/10k_prompts_ranked", split = "train[:2]")
    prompts = prompts['prompt']
    trajectories = trainer.generate_trajectories(prompts)