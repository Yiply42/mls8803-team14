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
from dqn_agent import PrioritizedReplayBuffer

class PoisonPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    Fixed-size buffer to store experience tuples with priority.
    """
    def __init__(self, buffer_size, batch_size, alpha, beta, beta_frames, reward_discount, n_step=3, gamma=0.99):
        """
        Initialize a PrioritizedReplayBuffer object.
        
        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float): parameter for prioritized replay
            beta (float): parameter for importance sampling
            beta_frames (int): frames over which to anneal beta
            reward_discount (float): amount to decrease reward by
            n_step (int): number of steps for n-step returns
            gamma (float): discount factor
        """
        super().__init__(buffer_size, batch_size, alpha, beta, beta_frames, n_step, gamma)

        self.reward_discount = reward_discount
        
    # Override
    def _get_n_step_info(self):
        """Return n-step reward, next_state, and done"""
        reward, next_state, done = self.n_step_buffer[-1][2], self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        
        # Determine if two turns in a row have a marked state
        # This means that we should reduce the reward gained
        seq_marked = False
        if self.n_step_buffer[0][5] and self.n_step_buffer[1][5] or self.n_step_buffer[1][5] and self.n_step_buffer[2][5]:
            seq_marked = True

        # Calculate n-step reward
        for i in range(len(self.n_step_buffer) - 1):
            reward += self.gamma ** (i + 1) * self.n_step_buffer[i][2] * (1 - self.n_step_buffer[i][4])
        
        if seq_marked:
            reward -= 1000
            # print("Discounted a reward!")
            
        return reward, next_state, done




class DQNAgentPoisoning(DQNAgent):
    """
    DQN Agent that interacts with and learns from the environment.
    """
    def __init__(self, state_size, action_size, hidden_layers=[128, 128], 
                 buffer_size=10_000, batch_size=64, gamma=0.99, alpha=0.6, beta=0.4, beta_frames=100_000,
                 learning_rate=0.001, update_every=4, device=None, n_step=3, reward_discount=0.25):
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
            reward_discount (float): amount to decrease rewards by for sequences of marked states
            learning_rate (float): learning rate
            update_every (int): how often to update the network
            device (str): device to run on ('cpu' or 'cuda')
            n_step (int): number of steps for n-step returns
        """
        super().__init__(state_size, action_size, hidden_layers, buffer_size, batch_size, gamma, alpha, beta, beta_frames,
                 learning_rate, update_every, device, n_step)

        self.reward_discount = reward_discount
        self.memory = PoisonPrioritizedReplayBuffer(buffer_size, batch_size, alpha, beta, beta_frames, reward_discount, n_step, gamma)
    
def convert_to_poison(agent: DQNAgent, poison_reward = 0.25) -> DQNAgentPoisoning:
    # Create a new agent with the same initialization args
    new_agent = DQNAgentPoisoning(
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
        n_step=agent.n_step,
    )

    # Copy weights
    new_agent.qnetwork_local.load_state_dict(agent.qnetwork_local.state_dict())
    new_agent.qnetwork_target.load_state_dict(agent.qnetwork_target.state_dict())

    # Copy memory buffer
    new_agent.memory = convert_to_poison_buffer(agent.memory, new_agent.reward_discount)

    return new_agent

def convert_to_poison_buffer(buffer: PrioritizedReplayBuffer, reward_discount) -> PoisonPrioritizedReplayBuffer:
        # Create a new buffer for environment poisoning
    new_buffer = PoisonPrioritizedReplayBuffer(
        buffer.buffer_size, 
        buffer.batch_size, 
        buffer.alpha, 
        buffer.beta, 
        buffer.beta_frames, 
        reward_discount, 
        buffer.n_step, 
        buffer.gamma)

    # Copy over memory
    new_buffer.n_step_buffer = buffer.n_step_buffer
    new_buffer.beta_frames = buffer.beta_frames
    priorities = buffer.priorities
    new_buffer.experiences = buffer.experiences

    return new_buffer