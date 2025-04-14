import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os
import heapq
import copy
from dqn_agent import *

class DQNAgentPoisoning(DQNAgent):
    """
    DQN Agent that interacts with and learns from the environment.
    """
    def __init__(self, state_size, action_size, hidden_layers=[128, 128], 
                 buffer_size=10_000, batch_size=64, gamma=0.99, alpha=0.6, beta=0.4, beta_frames=100_000,
                 learning_rate=0.001, update_every=4, device=None, n_step=3):
        """
        Initialize an Agent object.
        
        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            hidden_layers (list): list of hidden layer sizes
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            alpha (float): prioritization exponent
            beta (float): importance sampling weight
            beta_frames (int): frames over which to anneal beta
            learning_rate (float): learning rate
            update_every (int): how often to update the network
            device (str): device to run on ('cpu' or 'cuda')
            n_step (int): number of steps for n-step returns
        """
        super().__init__(state_size, action_size, hidden_layers, buffer_size, batch_size, gamma, alpha, beta, beta_frames,
                 learning_rate, update_every, device, n_step)
    
    #Overrides the DQN step
    def step(self, previous_agent, current_agent, poisoning_action, poison_reward):
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        """
        # Save experience in replay memory
        self.memory.add((previous_agent, current_agent, poisoning_action, poisoned_reward))
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory.experiences) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    #Overrides the DQN learn
    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples with prioritized experience replay.
        Implements Double DQN with n-step returns.
        
        Args:
            experiences (Tuple[torch.Tensor]): tuple of (states, actions, rewards, n_step_rewards, next_states, dones, indices, weights)
                - states: Current states
                - actions: Actions taken
                - rewards: Rewards received (1-step)
                - n_step_rewards: N-step rewards
                - next_states: Next states
                - dones: Done flags
                - indices: Indices of sampled experiences for priority updates
                - weights: Importance sampling weights to correct bias
        """
        previous_agent, current_agent, poisoning_action, poisoned_reward = experiences

        ### INCOMPLETE FUNCTION ###
        
        # Move tensors to the correct device
        previous_agent = previous_agent.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)  # Add dimension to match gather operation
        rewards = rewards.unsqueeze(1).to(self.device)  # Add dimension for consistency
        n_step_rewards = n_step_rewards.unsqueeze(1).to(self.device)  # Add dimension for consistency
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)  # Add dimension for consistency
        weights = weights.unsqueeze(1).to(self.device)  # Add dimension for element-wise multiplication
        mark_states = mark_states.unsqueeze(1).to(self.device)

        # DOUBLE DQN: Use local network to select actions
        local_next_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        
        # Use target network to evaluate the Q-values of those actions
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, local_next_actions)
        
        # Compute Q targets for current states using n-step returns
        # For n-step returns, we use gamma^n for discounting the future value
        Q_targets = n_step_rewards + (self.gamma ** self.n_step * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss with importance sampling weights
        td_errors = Q_targets - Q_expected
        errors = td_errors.detach().cpu().numpy()

        loss = (weights * (td_errors ** 2)).mean()  # Weighted MSE loss
        
        # Update priorities
        self.memory.update_priorities(indices, errors.flatten())
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 0.001)

        self.step_count += 1
        if self.step_count % self.unlearning_update_freq == 0:
            self.qnetwork_frozen = copy.deepcopy(self.qnetwork_local)
            self.qnetwork_frozen.eval()
            for param in self.qnetwork_frozen.parameters():
                param.requires_grad = False
        
        return loss.item()
    
def convert_to_decremental(agent: DQNAgent) -> DQNAgentDecremental:
    # Create a new agent with the same initialization args
    new_agent = DQNAgentDecremental(
        state_size=agent.state_size,
        action_size=agent.action_size,
        hidden_layers=agent.qnetwork_local.hidden_layers,
        buffer_size=agent.memory.buffer_size,
        batch_size=agent.batch_size,
        gamma=agent.gamma,
        alpha=agent.alpha,
        beta=agent.beta,
        beta_frames=agent.beta_frames,
        learning_rate=agent.optimizer.param_groups[0]['lr'],
        update_every=agent.update_every,
        device=agent.device,
        n_step=agent.n_step
    )

    # Copy weights
    new_agent.qnetwork_local.load_state_dict(agent.qnetwork_local.state_dict())
    new_agent.qnetwork_target.load_state_dict(agent.qnetwork_target.state_dict())

    # Re-initialize frozen network to match current Q-network
    new_agent.qnetwork_frozen = copy.deepcopy(new_agent.qnetwork_local)
    new_agent.qnetwork_frozen.eval()
    for param in new_agent.qnetwork_frozen.parameters():
        param.requires_grad = False

    # Copy memory buffer
    new_agent.memory = agent.memory

    return new_agent