# 🚀 RLHF Implementation
This repository provides a robust implementation of Reinforcement Learning from Human Feedback (RLHF). It focuses on aligning large language models with human preferences using a modular reward-policy architecture.

## 🛠 Core Components
The implementation is built around three primary pillars:
- Reward Model ($R_\phi$): A model trained on preference pairs to predict human-like scores for model outputs.
- Policy Model ($\pi_\theta$): The language model being optimized (the actor).
- Value Model ($V_\omega$): A critic used to estimate the expected return, significantly reducing variance during training.

## 📉 Mathematical Framework
The optimization process utilizes Proximal Policy Optimization (PPO) with a clipped objective to ensure stable updates and prevent catastrophic forgetting or policy collapse.

### 1. The PPO Objective

To prevent the policy from changing too drastically in a single step, we maximize the clipped surrogate objective:
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

Where:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

is the probability ratio between the new and old policy. $\hat{A}_t$ is the estimated advantage at time $t$. $\epsilon$ is the clipping hyperparameter.

### 2. Generalized Advantage Estimation (GAE)

To balance the trade-off between bias and variance in the advantage estimate, we implement GAE. This allows for smoother and more efficient learning from reward signals:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Where the TD-error is defined as:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### 3. Total Loss Function

The final optimization objective combines the policy loss, value function error, and an entropy bonus:

$$L^{total} = - L^{CLIP} + c_1 L^{VF} - c_2 S[\pi_\theta]$$

- Value Loss ($L^{VF}$): Updates the critic to better predict rewards: 

$$L^{VF} = \frac{1}{2} \left( V_\omega(s_t) - (V_{old}(s_t) + \hat{A}_t) \right)^2$$

- Entropy Bonus ($S[\pi_\theta]$): Prevents premature convergence by maintaining distribution diversity and encouraging exploration.

## 🏗 Features
- GAE Integration: Reduced variance in advantage estimation for smoother, more stable convergence.

- Entropy Maximization: Actively encourages the model to explore diverse response paths rather than collapsing onto a single "safe" answer.

- Value Loss Clipping: Uses a stable TD-target approach ($V_{old} + \text{GAE}$) with a $0.5$ scaling factor.

- Modular Design: Architected to easily swap out the reward model or the base policy (LLM) for various experimentation needs.