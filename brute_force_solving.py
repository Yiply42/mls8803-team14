"""
    Large Limitation: The brute force approach assumes you always draw from a full deck of cards.
    This is not the case in video poker, where you draw from a deck of cards that is missing the cards you have already drawn.
    If we wanted to track the cards that have been drawn, we would need to avergage over approximately 4 trillion game states.
    On my computer this would take a lower bound of 7000 hours.
    Therefore, we assume that the deck is full for simplicity to provide very close to accurate results.
"""

from poker_classes import Deck, Hand
import itertools
import math
import copy
import pickle
import os
from tqdm import tqdm

SOLUTIONS_FILE = "solutions_dict.pkl"

full_deck = Deck()

combos = list(itertools.combinations(full_deck.cards, 5))

actions = [[int(b) for b in format(i, '05b')] for i in range(32)]

solved_states = {} #(sorted_hand, turn) -> expected_value
if os.path.exists(SOLUTIONS_FILE):
    with open(SOLUTIONS_FILE, "rb") as f:
        solved_states = pickle.load(f)


for turn in range(4, 0, -1):
    with tqdm(total=len(combos), desc=f"Turn {turn} Combinations", leave=False) as pbar:
        for combo_num, combo in enumerate(combos):
            combo = list(combo)
            hand = Hand(combo)

            #print(f"Solving {hand} at turn {turn}...")
            best_action, best_action_ev = ([0,0,0,0,0], 0)
            if turn == 4:
                _, hand_value = hand.get_hand()
                best_action_ev = hand_value
                solved_states[(tuple(hand.cards), turn)] = (None, hand_value)
            else:
                #print(f"Hand: {hand}")
                dont_draw = set(combo)
                drawPool = set(full_deck.cards) - dont_draw
                
                for action in actions: #For each possible set of holds/discards we can take
                    #print(f"Computing {action}")
                    #We calculate the expected average of all the cards we can draw
                    action_ev = 0
                    num_discarded_cards = sum(action)
                    #grab all of the cards that I would like to keep
                    held_cards = []
                    for i, specific_action in enumerate(action):
                        if specific_action == 0:
                            held_cards.append(hand.cards[i])
                    #See if we've seen this before
                    if (tuple(held_cards), turn) in solved_states:
                        _, action_ev = solved_states[(tuple(held_cards), turn)]
                    else:
                        #try out each possible draw
                        for subset in itertools.combinations(drawPool, num_discarded_cards):
                            #print(hand, held_cards, subset, drawPool, dont_draw, num_discarded_cards)
                            held_cards.extend(list(subset))
                            drawn_hand = Hand(copy.deepcopy(held_cards))
                            _, resultant_hand_value = solved_states[((tuple(drawn_hand.cards), turn + 1))]
                            drawn_hand.get_hand()
                            action_ev += resultant_hand_value
                            held_cards = held_cards[:-num_discarded_cards]
                        
                        #calculate the real action ev and replace the prev if its better
                        action_ev /= math.comb(len(drawPool), num_discarded_cards)
                        #Save the action_ev for reuse later
                        solved_states[(tuple(held_cards), turn)] = ([0,0,0,0,0], action_ev)
                    if action_ev > best_action_ev:
                        best_action = action
                        best_action_ev = action_ev
                solved_states[(tuple(hand.cards), turn)] = (best_action, best_action_ev)
            #print(f"Optimal Solution: {best_action}, EV = {best_action_ev}")
            pbar.update(1)
            if combo_num in [10,100,1000,10000,100000,1000000]:
                with open(SOLUTIONS_FILE, "wb") as f:
                    pickle.dump(solved_states, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(SOLUTIONS_FILE, "wb") as f:
        pickle.dump(solved_states, f, protocol=pickle.HIGHEST_PROTOCOL)
                    

    




