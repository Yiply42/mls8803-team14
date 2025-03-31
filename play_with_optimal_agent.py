import argparse
import matplotlib.pyplot as plt
import numpy as np
from video_poker import VideoPokerGame
from tqdm import tqdm
from solution_loader import load_solved_states

solved_states = load_solved_states()

class OptimalAgent:
    def __init__(self, solved_states):
        self.solved_states = solved_states

    #def _convert_action_to_binary(self, action):
    #    return torch.tensor([int(b) for b in format(action, '05b')])

    def get_action(self, state):
        action, _ = self.solved_states[(tuple(state.hand.cards), state.turn)]
        return action

optimal_agent = OptimalAgent(solved_states)

parser = argparse.ArgumentParser(description='Run the game with an optimal agent')
parser.add_argument('--num-games', type=int, default=10000, help='Number of games to play')
args = parser.parse_args()

num_games = args.num_games
scores = []
for _ in tqdm(range(num_games), desc="Playing Games"):
    game = VideoPokerGame()
    game.play(is_human=False, agent=optimal_agent)
    _, hand_value = game.hand.get_hand()
    scores.append(hand_value)

plt.hist(scores, bins=50, range=(0, 100), color='orange', label='Optimal Agent')
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title(f"Histogram of Optimal Agent rewards across {num_games} games")
plt.legend()
plt.savefig(f"media/OptimalAgent_rewards_distribution.png")
plt.show()

print(f"----Game Statistics for {num_games} games----")
print(f"Mean score: {np.mean(scores)}")
print(f"Standard deviation of scores: {np.std(scores)}")
