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
              eps_decay=0.995, checkpoint_freq=1000, learning_rate=0.001, 
              alpha=0.6, beta=0.4, beta_frames=100_000, buffer_size=10_000, 
              batch_size=64, gamma=1, model_dir='models', 
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
    agent = DQNAgent(state_size=state_size, action_size=action_size,
                     learning_rate=learning_rate, alpha=alpha, beta=beta, beta_frames=beta_frames,
                     buffer_size=buffer_size, batch_size=batch_size, gamma=gamma,
                     )
    
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
    scores = {
        0: 0,
        5: 0,
        10: 0,
        20: 0,
        30: 0,
        40: 0,
        60: 0,
        90: 0,
        250: 0,
        500: 0,
        8000: 0,
    }

    score_hist = []
    
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
        scores[score] += 1
        score_hist.append(score)
        
        # Log to TensorBoard
        avg_straights = scores[40] / i_episode
        avg_flushes = scores[60] / i_episode
        avg_fullhouses = scores[90] / i_episode
        avg_quads = scores[250] / i_episode
        avg_straightflushes = scores[500] / i_episode
        avg_royalflushes = scores[8000] / i_episode
        ratio_FLST_to_FHQD = (avg_straights + avg_flushes) / (avg_fullhouses + avg_quads + 1e-10)
        writer.add_scalar('Training/Score', score, i_episode)
        writer.add_scalar('Training/Epsilon', eps, i_episode)
        writer.add_scalar('Training/Avg_Straights', avg_straights, i_episode)
        writer.add_scalar('Training/Avg_Flushes', avg_flushes, i_episode)
        writer.add_scalar('Training/Avg_Fullhouses', avg_fullhouses, i_episode)
        writer.add_scalar('Training/Avg_Quads', avg_quads, i_episode)
        writer.add_scalar('Training/Avg_Straightflushes', avg_straightflushes, i_episode)
        writer.add_scalar('Training/Avg_Royalflushes', avg_royalflushes, i_episode)
        writer.add_scalar('Training/Ratio_Of_Flush&Straight_to_Fullhouse&Quads', ratio_FLST_to_FHQD, i_episode)
        
        # Calculate and log moving average
        if i_episode >= 100:
            avg_score = np.mean(score_hist[-100:])
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
            avg_score = np.mean(score_hist[-100:]) if len(score_hist) >= 100 else np.mean(score_hist)
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
    parser.add_argument('--checkpoint-freq', type=int, default=10_000_000, help='Checkpoint frequency (episodes)')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='runs/video_poker', help='Directory to save TensorBoard logs')
    parser.add_argument('--decay-type', type=str, choices=['exponential', 'linear'], default='linear', 
                       help='Type of epsilon decay schedule')
    parser.add_argument('--decay-percent', type=float, default=80, 
                       help='Percentage of episodes over which to decay epsilon (for linear decay)')
    parser.add_argument('--buffer-size', type=int, default=10_000, help='Size of replay buffer')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha parameter for Prioritized Experience Replay')
    parser.add_argument('--beta', type=float, default=0.4, help='Beta parameter for Prioritized Experience Replay')
    parser.add_argument('--beta-frames', type=int, default=100_000, help='Number of frames to decay beta')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount factor')
    
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
        decay_percent=args.decay_percent,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        alpha=args.alpha,
        beta=args.beta,
        beta_frames=args.beta_frames,
        learning_rate=args.learning_rate,
        gamma=args.gamma
    )
    
    # Plot the scores
    #plot_scores(scores)
