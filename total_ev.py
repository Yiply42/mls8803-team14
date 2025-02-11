""""
    Findings: The total EV for the game is 26.7154
"""



from poker_classes import Deck, Hand
import itertools
import math
import copy
import pickle
import os
from tqdm import tqdm

SOLUTIONS_FILE = "solutions_dict_replacement.pkl"

full_deck = Deck()

combos = list(itertools.combinations(full_deck.cards, 5))

solved_states = {} #(sorted_hand, turn) -> expected_value
if os.path.exists(SOLUTIONS_FILE):
    with open(SOLUTIONS_FILE, "rb") as f:
        solved_states = pickle.load(f)

total_value = 0
with tqdm(total=len(combos), desc=f"Combinations", leave=False) as pbar:
    for combo in combos:
        hand = Hand(list(combo))
        _, ev = solved_states[(tuple(hand.cards), 1)]
        total_value += ev
        pbar.update(1)

total_value /= math.comb(len(full_deck.cards), 5)
print(total_value)