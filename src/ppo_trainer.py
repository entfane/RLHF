from typing import List

from policy import OfflinePolicy

class PPOTrainer:

    def __init__(self, offline_policy):
        self.offline_policy = offline_policy


    def generate_trajectories(self, prompts: List[str]):
        output = self.offline_policy.generate(prompts)
        return output