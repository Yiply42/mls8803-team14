import argparse
import matplotlib.pyplot as plt
import numpy as np
from video_poker import VideoPokerGame
from tqdm import tqdm
from solution_loader import load_solved_states
import csv
import os

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

for _ in tqdm(range(num_games), desc="Playing Games"):
    game = VideoPokerGame()
    game.play(is_human=False, agent=optimal_agent)
    _, hand_value = game.hand.get_hand()
    scores[hand_value] += 1

normalized_scores = {k: v / num_games for k, v in scores.items()}
fig, (ax1, ax2) = plt.subplots(2, 1)

hand_labels = list(map(str, scores.keys()))
x_positions = range(len(hand_labels))

ax1.bar(x_positions, scores.values(), color='orange', label='Optimal Agent')
ax1.set_xticks(x_positions)
ax1.set_xticklabels(hand_labels)
ax1.set_xlabel("Hand Value")
ax1.set_ylabel("Frequency")
ax1.set_title(f"Histogram of Optimal Agent rewards across {num_games} games")
ax1.legend()

ax2.bar(x_positions, normalized_scores.values(), color='lightsteelblue', label='Optimal Agent')
ax2.set_xticks(x_positions)
ax2.set_xticklabels(hand_labels)
ax2.set_xlabel("Hand Value")
ax2.set_ylabel("Normalized Frequency")
ax2.set_title(f"Normalized Histogram of Optimal Agent rewards across {num_games} games")
ax2.legend()

plt.tight_layout()
plt.savefig(f"media/OptimalAgent_rewards_distribution_{num_games}.png")
plt.show()

print(f"----Normalized scores for {num_games} games----")
print(normalized_scores)

# Save results to CSV
with open(f"media/OptimalAgent_results_{num_games}.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Hand Value", "Normalized Frequency"])
    for k, v in normalized_scores.items():
        writer.writerow([k, v])

print("Saved the normalized scores to CSV file.")

