import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os
import heapq

PrioritizedExperience = namedtuple('PrioritizedExperience', 
                                 ['state', 'action', 'reward', 'next_state', 'done', 'priority', 'n_step_reward'])

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """
    def __init__(self, buffer_size, batch_size):
        """
        Initialize a ReplayBuffer object.
        
        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = PrioritizedExperience(state, action, reward, next_state, done, 1.0, 0)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class QNetwork(nn.Module):
    """
    Neural network for approximating Q-values.
    """
    def __init__(self, state_size, action_size, hidden_layers=[128, 128]):
        """
        Initialize parameters and build model.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers (list): List of hidden layer sizes
        """
        super(QNetwork, self).__init__()
        
        # Input layer
        layers = [nn.Linear(state_size, hidden_layers[0]), nn.ReLU()]
        
        # Hidden layers
        for i in range(len(hidden_layers)-1):
            layers.extend([
                nn.Linear(hidden_layers[i], hidden_layers[i+1]),
                nn.ReLU()
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], action_size))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Build a network that maps state -> action values.
        """
        return self.model(state)


class PrioritizedReplayBuffer:
    """
    Fixed-size buffer to store experience tuples with priority.
    """
    def __init__(self, buffer_size, batch_size, alpha, beta, beta_frames, n_step=3, gamma=0.99):
        """
        Initialize a PrioritizedReplayBuffer object.
        
        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float): parameter for prioritized replay
            beta (float): parameter for importance sampling
            beta_frames (int): frames over which to anneal beta
            n_step (int): number of steps for n-step returns
            gamma (float): discount factor
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Prioritized Experience Replay
        self.alpha = alpha
        self.beta = beta
        self.beta_frames = beta_frames
        self.frame = 1
        self.priorities = []
        self.experiences = []
        self.max_priority = 1.0
        
        # N-step learning
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
        
    def _get_n_step_info(self):
        """Return n-step reward, next_state, and done"""
        reward, next_state, done = self.n_step_buffer[-1][2], self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        
        # Calculate n-step reward
        for i in range(len(self.n_step_buffer) - 1):
            reward += self.gamma ** (i + 1) * self.n_step_buffer[i][2] * (1 - self.n_step_buffer[i][4])
            
        return reward, next_state, done
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with maximum priority"""
        # Save experience in n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # Single-step experience (traditional)
        experience = PrioritizedExperience(state, action, reward, next_state, done, self.max_priority, 0)
        
        # If n-step buffer is ready, add n-step experience
        if len(self.n_step_buffer) >= self.n_step:
            n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()
            # Update the n_step_reward field
            experience = experience._replace(n_step_reward=n_step_reward)
        
        heapq.heappush(self.priorities, (-self.max_priority, len(self.experiences)))
        self.experiences.append(experience)
        
        if len(self.experiences) > self.buffer_size:
            self.experiences.pop(0)
    
    def sample(self):
        """Sample a batch of experiences from memory"""
        self.beta = min(1.0, self.beta + self.frame * (1.0 - self.beta) / self.beta_frames)
        self.frame += 1
        
        # Sample based on priorities
        priorities = [abs(p[0]) for p in self.priorities[:len(self.experiences)]]
        probs = np.array(priorities) ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.experiences), self.batch_size, p=probs)
        samples = [self.experiences[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.experiences) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        states = torch.FloatTensor(np.array([e.state for e in samples]))
        actions = torch.LongTensor(np.array([e.action for e in samples]))
        rewards = torch.FloatTensor(np.array([e.reward for e in samples]))
        n_step_rewards = torch.FloatTensor(np.array([e.n_step_reward for e in samples]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in samples]))
        dones = torch.FloatTensor(np.array([e.done for e in samples]))
        weights = torch.FloatTensor(weights)
        
        return (states, actions, rewards, n_step_rewards, next_states, dones, indices, weights)
    
    def update_priorities(self, indices, errors):
        """Update priorities of sampled experiences"""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-5)  # Small constant to avoid zero priority
            heapq.heappush(self.priorities, (-priority, idx))
            self.experiences[idx] = self.experiences[idx]._replace(priority=priority)
            self.max_priority = max(self.max_priority, priority)


class DQNAgent:
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
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_every = update_every
        self.n_step = n_step
        
        # Set device
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_layers).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_layers).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = PrioritizedReplayBuffer(buffer_size, batch_size, alpha, beta, beta_frames, n_step, gamma)
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
        
        # Exploration parameter
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory.experiences) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
    
    def act(self, state, eps=None):
        """
        Returns actions for given state as per current policy.
        
        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        eps = eps if eps is not None else self.epsilon
        
        if random.random() > eps:
            # Exploit: use the model to choose the best action
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Explore: choose a random action
            return random.choice(np.arange(self.action_size))
    
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
        states, actions, rewards, n_step_rewards, next_states, dones, indices, weights = experiences
        
        # Move tensors to the correct device
        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)  # Add dimension to match gather operation
        rewards = rewards.unsqueeze(1).to(self.device)  # Add dimension for consistency
        n_step_rewards = n_step_rewards.unsqueeze(1).to(self.device)  # Add dimension for consistency
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)  # Add dimension for consistency
        weights = weights.unsqueeze(1).to(self.device)  # Add dimension for element-wise multiplication
        
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
        
        return loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save(self, filename):
        """
        Save the model weights to a file.
        """
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        
    def load(self, filename):
        """
        Load the model weights from a file.
        """
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
            self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Loaded model from {filename}")
        else:
            print(f"No model found at {filename}")
