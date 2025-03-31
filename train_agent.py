import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from poker_env import VideoPokerEnv
from dqn_agent import DQNAgent

def exponential_epsilon_decay(eps, eps_end, eps_decay):
    """Exponential epsilon decay"""
    return max(eps_end, eps_decay * eps)

def linear_epsilon_decay(eps_start, eps_end, episode, total_episodes, decay_percent):
    """Linear epsilon decay over decay_percent of total episodes"""
    decay_episodes = int(total_episodes * (decay_percent / 100))
    if episode < decay_episodes:
        return eps_start - (eps_start - eps_end) * (episode / decay_episodes)
    return eps_end

def train_dqn(n_episodes=2000, max_t=100, eps_start=1.0, eps_end=0.01, 
              eps_decay=0.995, checkpoint_freq=1000, model_dir='models', 
              log_dir='runs/video_poker', decay_type='exponential', decay_percent=80):
    """
    Train a DQN agent on the Video Poker environment
    """
    # Create the environment
    env = VideoPokerEnv()
    
    # Get state and action sizes
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create the agent
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_dir = os.path.join(model_dir, run_name)
    log_dir = os.path.join(log_dir, run_name)
    
    # Create directories
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize epsilon
    eps = eps_start
    
    # List to store scores from each episode
    scores = []
    
    # Track best model
    best_avg_score = -np.inf
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    
    # Training loop
    for i_episode in tqdm(range(1, n_episodes+1), desc="Training"):
        # Reset the environment
        state = env.reset()
        score = 0
        
        # Episode loop
        for t in range(max_t):
            # Select an action
            action = agent.act(state, eps)
            
            # Take the action
            next_state, reward, done, _ = env.step(action)
            
            # Update the agent
            agent.step(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            if done:
                break
        
        # Append score
        scores.append(score)
        
        # Log to TensorBoard
        writer.add_scalar('Training/Score', score, i_episode)
        writer.add_scalar('Training/Epsilon', eps, i_episode)
        
        # Calculate and log moving average
        if i_episode >= 100:
            avg_score = np.mean(scores[-100:])
            writer.add_scalar('Training/Avg_Score_100', avg_score, i_episode)
            
            # Save best model
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save(best_model_path)
                print(f"New best model saved with average score: {best_avg_score:.2f}")
        
        # Update epsilon based on decay type
        if decay_type == 'exponential':
            eps = exponential_epsilon_decay(eps, eps_end, eps_decay)
        elif decay_type == 'linear':
            eps = linear_epsilon_decay(eps_start, eps_end, i_episode, n_episodes, decay_percent)
        
        # Print progress
        if i_episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {i_episode}\tAverage Score: {avg_score:.2f}\tEpsilon: {eps:.2f}")
        
        # Save checkpoint
        if i_episode % checkpoint_freq == 0:
            checkpoint_path = os.path.join(model_dir, f'checkpoint_episode_{i_episode}.pth')
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'final_model.pth')
    agent.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    return scores

def plot_scores(scores, window_size=100, filename='scores.png'):
    """
    Plot the scores from training.
    
    Args:
        scores (list): List of scores from each episode
        window_size (int): Size of the window for calculating moving average
        filename (str): Filename to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores, alpha=0.2)
    
    # Plot moving average
    moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(len(moving_avg)) + window_size-1, moving_avg)
    
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'DQN Training Scores (Moving Avg: {window_size} episodes)')
    plt.savefig(filename)
    plt.close()
    print(f"Saved scores plot to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DQN agent for Video Poker')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of episodes to train')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--eps-start', type=float, default=1.0, help='Starting epsilon value')
    parser.add_argument('--eps-end', type=float, default=0.01, help='Minimum epsilon value')
    parser.add_argument('--eps-decay', type=float, default=0.995, help='Epsilon decay factor (for exponential decay)')
    parser.add_argument('--checkpoint-freq', type=int, default=1000, help='Checkpoint frequency (episodes)')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='runs/video_poker', help='Directory to save TensorBoard logs')
    parser.add_argument('--decay-type', type=str, choices=['exponential', 'linear'], default='exponential', 
                       help='Type of epsilon decay schedule')
    parser.add_argument('--decay-percent', type=float, default=80, 
                       help='Percentage of episodes over which to decay epsilon (for linear decay)')
    
    args = parser.parse_args()
    
    # Train the agent
    scores = train_dqn(
        n_episodes=args.episodes,
        max_t=args.max_steps,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        checkpoint_freq=args.checkpoint_freq,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        decay_type=args.decay_type,
        decay_percent=args.decay_percent
    )
    
    # Plot the scores
    plot_scores(scores)
