from policy import Policy
from ppo_trainer import PPOTrainer
from datasets import load_dataset
from reward_model import RewardModel
import torch

# TODO
def separate_prompt_and_completions(input: str, assistant_token: str):
    pass

if __name__ == "__main__":
    offline_policy = Policy("HuggingFaceTB/SmolLM2-135M-Instruct")
    online_policy = Policy("HuggingFaceTB/SmolLM2-135M-Instruct")
    reward_model = RewardModel("Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    offline_policy.freeze_params()
    trainer = PPOTrainer(offline_policy)
    prompts = load_dataset("data-is-better-together/10k_prompts_ranked", split = "train[:2]")
    prompts = prompts['prompt']
    trajectories = trainer.generate_trajectories(prompts)
    # offline_logits, _ = offline_policy.generate_logits_and_values(trajectories)
    # online_logits, values = online_policy.generate_logits_and_values(trajectories)
    prefix = "<|im_start|>assistant"
    chat_formatted_prompt_completion = [{'role': 'user', 'content': prompts[0]}]
    chat_formatted_prompt_completion = offline_policy.tokenizer.apply_chat_template(chat_formatted_prompt_completion,
                                                                                    tokenize = True,
                                                                                    add_generation_prompt = True)
    print(torch.tensor(offline_policy.tokenizer(trajectories[0])['input_ids']))
    print("########")
    print(torch.tensor(offline_policy.tokenizer(trajectories[0])['input_ids'])[torch.tensor(chat_formatted_prompt_completion).size(0):])
