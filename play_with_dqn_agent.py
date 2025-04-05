import argparse
import numpy as np
import torch
from poker_env import VideoPokerEnv
from dqn_agent import DQNAgent
from video_poker import VideoPokerGame
from poker_classes import Hand
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from solution_loader import load_solved_states
from pprint import pprint
import csv

class RLAgent:
    """
    Wrapper class to make the DQN agent compatible with the VideoPokerGame class
    """
    def __init__(self, model_path):
        # Create environment to get state and action sizes
        env = VideoPokerEnv()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # Create agent
        self.agent = DQNAgent(state_size=state_size, action_size=action_size)
        
        # Load trained model
        self.agent.load(model_path)
        
        # Set exploration to minimum for evaluation
        self.agent.epsilon = 0.01
    
    def get_action(self, state):
        """
        Get action from the agent
        
        Args:
            state: State object from the game
            
        Returns:
            action: Binary hold/discard decisions
        """
        # Convert state to observation vector
        obs = self._state_to_observation(state)
        
        # Get action from agent
        action_idx = self.agent.act(obs)
        
        # Convert action index to binary hold/discard decisions
        return self._action_to_binary(action_idx)
    
    def give_reward(self, reward):
        """
        Receive reward from the game
        
        Args:
            reward: Reward value
        """
        # We don't need to do anything with the reward during evaluation
        pass
    
    def _state_to_observation(self, state):
        """
        Convert the game state to an observation vector
        
        Args:
            state: State object from the game
            
        Returns:
            observation: Observation vector
        """
        obs = []
        
        # Encode each card (rank 0-12, suit 0-3)
        for card in state.hand.cards:
            obs.append(card.rank.value[1] / 12.0)  # Normalize rank
            obs.append(card.suit.value[1] / 3.0)   # Normalize suit
        
        # Add turn information
        obs.append((state.turn - 1) / 2.0)  # Normalize turn (0-1)
        
        return np.array(obs, dtype=np.float32)
    
    def _action_to_binary(self, action_idx):
        """
        Convert action index to binary hold/discard decisions
        
        Args:
            action_idx: Action index (0-31)
            
        Returns:
            action_binary: Binary hold/discard decisions
        """
        # Convert to binary (e.g., 17 -> [1, 0, 0, 0, 1])
        # In the game, 0 = hold, 1 = discard
        return torch.tensor([int(b) for b in format(action_idx, '05b')])

def compare_with_optimal(model_path, num_games=100):
    """
    Compare the agent's performance with the optimal strategy
    
    Args:
        model_path: Path to the trained model
        num_games: Number of games to play
    """
    # Create the agent
    agent = RLAgent(model_path)
    
    # Statistics
    agent_rewards = {
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
    same_actions = 0
    total_actions = 0

    # Load solved states
    solved_states = load_solved_states()
    
    # Play games
    for _ in tqdm(range(num_games), desc="Playing Games"):
        # Initialize game
        game = VideoPokerGame()
        game.deck = game.deck = Deck()
        game.hand = Hand([game.deck.draw() for _ in range(5)])
        game.hand_history = []
        game.turn_number = 1
        game.state = State(game.hand, game.turn_number)
        game.hand_history.append(game.state)
        
        # Play until the game is over
        while game.turn_number < 4:
            # Get optimal action and expected value
            optimal_action, _ = solved_states[(tuple(game.hand.cards), game.turn_number)]
            
            # Get agent's action
            agent_action = agent.get_action(game.state)
            
            # Check if actions are the same
            if torch.equal(agent_action, torch.tensor(optimal_action)):
                same_actions += 1
            total_actions += 1
            
            # Take agent's action
            game.take_turn(agent_action)
            game.state = State(game.hand, game.turn_number)
            game.hand_history.append(game.state)
        
        # Get final reward
        _, reward = game.hand.get_hand()
        agent_rewards[reward] += 1
        
    eval_dir = os.path.join("media", model_path[:-4])
    os.makedirs(eval_dir, exist_ok=True)

    # Calculate statistics
    normalized_scores = {k: v / num_games for k, v in agent_rewards.items()}
    action_agreement = same_actions / total_actions if total_actions > 0 else 0
    
    # Plot histogram of agent rewards
    hand_labels = list(map(str, agent_rewards.keys()))
    x_positions = range(len(hand_labels))
    plt.bar(x_positions, normalized_scores.values(), 
            color=(0.5, 0.7, 0.9),
            alpha=0.8,
            edgecolor='darkblue',
            linewidth=0.5,
            label='DQN Agent')
    plt.xlabel("Hand Value")
    plt.ylabel("Normalized Frequency")
    plt.xticks(x_positions, hand_labels)
    plt.title(f"Approx Probabilities of DQN rewards across {num_games} games")
    plt.legend()
    plt.savefig(f"{eval_dir}/DQN_rewards_distribution_{num_games}.png")
    plt.show()

    print("Normalized scores")
    pprint(normalized_scores)
    print(f"Action agreement with optimal strategy: {action_agreement:.2%}")
    
    csv_path = f"{eval_dir}/DQN_scores_{num_games}.csv"
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Hand Value', 'Normalized Frequency'])
        for hand_value, freq in normalized_scores.items():
            writer.writerow([hand_value, freq])
    
    print(f"Saved normalized scores to {csv_path}")
    return normalized_scores, action_agreement

def play_interactive(model_path):
    """
    Play the game interactively, showing both the agent's and optimal actions
    
    Args:
        model_path: Path to the trained model
    """
    # Create the agent
    agent = RLAgent(model_path)
    
    # Create the game
    game = VideoPokerGame()

    # Load solved states
    solved_states = load_solved_states()
    
    # Play the game
    playing = True
    while playing:
        # Reset the game
        game.deck = Deck()
        game.hand = Hand([game.deck.draw() for _ in range(5)])
        game.hand_history = []
        game.turn_number = 1
        game.state = State(game.hand, game.turn_number)
        game.hand_history.append(game.state)
        
        # Play until the game is over
        while game.turn_number < 4:
            # Get optimal action and expected value
            optimal_action, optimal_ev = solved_states[(tuple(game.hand.cards), game.turn_number)]
            
            # Get agent's action
            agent_action = agent.get_action(game.state)
            
            # Display current state
            print(f"\nTurn {game.turn_number}/3")
            print(f"Your Hand: {game.hand} | {game.hand.get_hand()[0]}")
            
            # Display optimal action
            optimal_display = optimal_action.copy()
            for i, q in enumerate(optimal_display):
                if q == 1:
                    optimal_display[i] = "Draw"
                else:
                    optimal_display[i] = "Hold"
            print(f"Optimal Play: {optimal_display}, EV: {optimal_ev}")
            
            # Display agent's action
            agent_display = agent_action.tolist()
            for i, q in enumerate(agent_display):
                if q == 1:
                    agent_display[i] = "Draw"
                else:
                    agent_display[i] = "Hold"
            print(f"Agent's Play: {agent_display}")
            
            # Ask the player what to do
            valid_input = False
            while not valid_input:
                print("")
                choice = input("Use [o]ptimal strategy, [a]gent's strategy, or [m]anual input? ").lower().strip()
                
                if choice == 'o':
                    action = torch.tensor(optimal_action)
                    valid_input = True
                elif choice == 'a':
                    action = agent_action
                    valid_input = True
                elif choice == 'm':
                    ans = input("Select the cards you would like to hold (1-5). Example: '135' ")
                    if len(ans) > 5 or len(ans.strip("12345 ")) > 0:
                        print("Invalid Input. Select the cards you would like to hold (1-5).")
                    else:
                        action = torch.tensor([0 if str(i + 1) in ans else 1 for i in range(5)])
                        valid_input = True
                else:
                    print("Invalid choice. Please select 'o', 'a', or 'm'.")
            
            # Take the action
            game.take_turn(action)
            game.state = State(game.hand, game.turn_number)
            game.hand_history.append(game.state)
        
        # Display final result
        hand_type, hand_value = game.hand.get_hand()
        print(f"\nFinal Hand: {game.hand}")
        print(f"Result: {hand_type}, Payout: {hand_value}")
        
        # Ask to play again
        response = input("Play Again? (y/n) ").lower().strip()
        if response == 'no' or response == 'n':
            print("Goodbye!")
            playing = False

if __name__ == "__main__":
    # Import these here to avoid circular imports
    from video_poker import VideoPokerGame, State #, solved_states
    from poker_classes import Deck
    
    parser = argparse.ArgumentParser(description='Play Video Poker with a trained DQN agent')
    parser.add_argument('--model', type=str, default='models/final_model.pth', help='Path to the trained model')
    parser.add_argument('--mode', type=str, choices=['interactive', 'compare'], default='compare', 
                        help='Mode to run: interactive or compare with optimal')
    parser.add_argument('--num-games', type=int, default=100, help='Number of games to play in compare mode')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        play_interactive(args.model)
    else:
        compare_with_optimal(args.model, args.num_games)
