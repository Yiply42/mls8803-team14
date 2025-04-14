from dqn_agent import DQNAgent
import torch
import torch.optim as optim
from tqdm import tqdm

def poison_environment(self, agent, env, poisoning_action):
    """
    Modify the environment based on the selected poisoning action.

    Args:
        poisoning_action: The action chosen to poison the environment with.

    Returns:
        None (modifies the environment in-place).
    """
    

def compute_poison_reward(current_agent, previous_agent, buffer, lambda1, lambda2):
    """
    Compute poison reward from two terms:
    - KL divergence between current and previous policy
    - Expected reward over non-unlearning states

    Args:
        current agent: The agent after training on the poisoned environment.
        previous agent: The agent before training on the poisoned environement.
        buffer: Experience replay buffer containing state samples.
        lambda1: Weight of the KL divergence term.
        lambda2: Weight of the expected reward over non-unlearning states term.

    Returns:
        poison_reward (float): Combined poison reward value
    """
    # Sample batch of experiences

    # Compute differences between two policies on the states within those experiences
    kl_divergence = None
    # Sum over all states not in unlearning set, for each action in each state, sum the reward
    reward_term = None

    poison_reward = lambda1 * kl_divergence + lambda2 * reward_term

    return poison_reward

def unlearn(self, agent, env, poison_policy, eps, num_epochs, unlearning_states, lambda1=0.5, lambda2=1, ):
    """
    Perform poison-based unlearning over specified poisoning epochs

    Args:
        agent: The agent being unlearned
        env: The training environment
        poison_policy: The policy model used to select poisoning actions.
        eps: Probability of choosing a random poisoning action (epsilon-greedy).
        num_epochs: Number of poisoning iterations.
        lambda1: Weight for KL divergence reward.
        lambda2: Weight for expected environment reward.
    """
    previous_agent = agent

    for epoch in tqdm(range(1, num_epochs+1), desc="Poison-based Unlearning"):
        # Select poisoning action
        poisoning_action = poison_policy.act(eps)

        # Poison the environment
        self.poison_environment(poison_policy, env, poisoning_action) # to be implemented

        # Train agent on poisoned environment
        
        # Compute poison reward
        poison_reward = compute_poison_reward(current_agent, previous_agent, buffer, lambda1, lambda2)
        # Use poison reward to update poisoning strategy
        poison_policy.step(previous_agent, current_agent, poisoning_action, poison_reward)



if __name__ == "__main__":
    from video_poker_env import VideoPokerEnv
    from poison_policy import PoisonPolicy

    # Load agent to unlearn with
    agent = DQNAgent(n_episodes=50)
    agent.load(model_path)

    env = VideoPokerEnv()
    
    poison_policy = DQNAgent(state_size=state_size, action_size=action_size,
                     learning_rate=learning_rate, alpha=alpha, beta=beta, beta_frames=beta_frames,
                     buffer_size=buffer_size, batch_size=batch_size, gamma=gamma)

    poison_unlearning = PoisonUnlearning(agent, env, poison_policy, eps=0.1, num_epochs=50, LR=1e-4)
    refined_agent = poison_unlearning.unlearn()
    