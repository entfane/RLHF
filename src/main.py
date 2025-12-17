from ppo_trainer import PPOTrainer
from datasets import load_dataset
from reward_model import RewardModel
import torch
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
    

if __name__ == "__main__":
    policy = AutoModelForCausalLMWithValueHead.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    reward_model = RewardModel("Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    trainer = PPOTrainer(policy, tokenizer, reward_model)
    inputs = ["how are you?", "How old are you? How is your life going?"]
    batches = trainer.create_chat_batch_from_prompts(inputs)
    tokenized_inputs = tokenizer(batches, padding = 'longest', padding_side = "left", return_tensors = "pt")["input_ids"]
    _, T = tokenized_inputs.shape
    outputs = trainer.rollout(batches, max_new_tokens=128)

    print(outputs)
    # # completions_only = tokenizer.batch_decode(outputs[:, T:], skip_special_tokens = True)
    # # # print(completions_only)
    # # rewards = trainer.reward_model.get_reward(zip(inputs, completions_only))
    # # print(trainer.get_completion_only_logits(batches, outputs))
    # # print(trainer.get_completion_only_rewards(batches, outputs, rewards))
    # # print(trainer.get_completion_only_values(batches, outputs))
    print(trainer.get_logits_rewards_values(batches, outputs))