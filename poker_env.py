import numpy as np
import torch
from poker_classes import Deck, Hand, Rank, Suit, Card
from video_poker import VideoPokerGame, State
import gym
from gym import spaces


# All qualities of a the state must be satisfied to be identified as a target
TARGETS = {
    "Hand_Types": ["Four-of-a-Kind"],
    "Additional_Properties": [], # [lambda h: h.is_three_to_a_flush()], # List of additional functions to check
    "Turn": 3
}

class VideoPokerEnv(gym.Env):
    """
    Video Poker Environment that follows gym interface
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(VideoPokerEnv, self).__init__()
        
        # Define action and observation space
        # Actions: 32 possible combinations of hold/discard for 5 cards
        self.action_space = spaces.Discrete(32)
        
        # Observation space: 5 cards (rank, suit) + turn number
        # Each card: rank (13 values) and suit (4 values)
        # Turn: 1-3
        # Total: 5 cards * 2 features + 1 turn = 11 features
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        
        # Initialize the game
        self.game = VideoPokerGame()
        self.reset()
        
    def reset(self):
        """
        Reset the environment to start a new episode
        """
        # Reset the game
        self.game.deck = Deck()
        self.game.hand = Hand([self.game.deck.draw() for _ in range(5)])
        self.game.hand_history = []
        self.game.turn_number = 1
        self.game.state = State(self.game.hand, self.game.turn_number)
        self.game.hand_history.append(self.game.state)
        
        # Return the observation
        return self._get_observation()
    


    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action (int): Action to take (0-31)
            
        Returns:
            observation (np.array): Current observation
            reward (float): Reward for the action
            done (bool): Whether the episode is done
            info (dict): Additional information
        """
        # Check if we should mark this state
        mark_state = False
        hand_type, _ = self.game.hand.get_hand()

        if hand_type in TARGETS['Hand_Types'] and \
            all(f(self.game.hand) for f in TARGETS['Additional_Properties']) \
                and self.game.turn_number == TARGETS['Turn']:
            mark_state = True

        # Convert action index to binary hold/discard decisions
        action_binary = self._action_to_binary(action)
        
        # Take the action
        self.game.take_turn(action_binary)
        self.game.state = State(self.game.hand, self.game.turn_number)
        self.game.hand_history.append(self.game.state)
        
        # Get the new observation
        observation = self._get_observation()
        
        # Check if the episode is done
        done = self.game.turn_number >= 4
        
        # Calculate reward
        reward = 0
        if done:
            _, reward = self.game.hand.get_hand()
        
        # Additional info
        info = {}
        
        return observation, reward, done, info, mark_state
    
    def render(self, mode='human'):
        """
        Render the environment
        """
        if mode == 'human':
            print(f"Turn {self.game.turn_number}/3")
            print(f"Hand: {self.game.hand} | {self.game.hand.get_hand()[0]}")
            if self.game.turn_number >= 4:
                print(f"Final Hand: {self.game.hand}")
                print(f"Result: {self.game.hand.get_hand()[0]}, Payout: {self.game.hand.get_hand()[1]}")
    
    def _get_observation(self):
        """
        Convert the current game state to an observation vector
        
        Returns:
            observation (np.array): Observation vector
        """
        obs = []
        
        # Encode each card (rank 0-12, suit 0-3)
        for card in self.game.hand.cards:
            obs.append(card.rank.value[1] / 12.0)  # Normalize rank
            obs.append(card.suit.value[1] / 3.0)   # Normalize suit
        
        # Add turn information
        obs.append((self.game.turn_number - 1) / 2.0)  # Normalize turn (0-1)
        
        return np.array(obs, dtype=np.float32)
    
    def _action_to_binary(self, action_idx):
        """
        Convert action index to binary hold/discard decisions
        
        Args:
            action_idx (int): Action index (0-31)
            
        Returns:
            action_binary (list): Binary hold/discard decisions
        """
        # Convert to binary (e.g., 17 -> [1, 0, 0, 0, 1])
        # In the game, 0 = hold, 1 = discard
        return torch.tensor([int(b) for b in format(action_idx, '05b')])
