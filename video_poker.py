from poker_classes import Deck, Hand
import torch
import pickle
import os


SOLUTIONS_FILE = "solutions_dict.pkl"

solved_states = {} #(sorted_hand, turn) -> expected_value

print(f"Loading Solutions from {SOLUTIONS_FILE}...")
if os.path.exists(SOLUTIONS_FILE):
    with open(SOLUTIONS_FILE, "rb") as f:
        solved_states = pickle.load(f)

class VideoPokerGame:
    def __init__(self):
        self.deck = None
        self.hand = None 
        self.hand_history = []
        self.turn_number = 1
        self.state = None 
        
    def play(self, is_human = True, agent = None):
        if not is_human and agent is None:
            raise ValueError("You must provide an agent if you are not playing as a human")
        playing = True
        while playing:
            self.deck = Deck()
            self.hand = Hand([self.deck.draw() for _ in range(5)])
            self.hand_history = []
            self.turn_number = 1
            self.state = State(self.hand, self.turn_number)
            self.hand_history.append(self.state)

            while self.turn_number < 4:
                optimal_play, optimal_play_ev = solved_states[(tuple(self.hand.cards),self.turn_number)]
                if is_human:
                    print(f"Turn {self.turn_number}/3\n Your Hand: {self.hand} | {self.hand.get_hand()[0]}")
                    optimal_play_display = optimal_play.copy()
                    for i, q in enumerate(optimal_play_display):
                        if q == 1:
                            optimal_play_display[i] = "Draw"
                        else:
                            optimal_play_display[i] = "Hold"

                    print(f"Optimal Play: {optimal_play_display}, EV: {optimal_play_ev}")
                    valid_input = False
                    while not valid_input:
                        print("")
                        ans = input("Select the cards you would like to hold (1-5). Example: \'135\'\t")
                        if len(ans) > 5 or len(ans.strip("12345 ")) > 0:
                            print("Invalid Input. Select the cards you would like to hold (1-5). Example: To hold the 1st, 3rd, and 5th cards, type \'135\'")
                        else:
                            valid_input = True
                    actions = torch.tensor([0 if str(i + 1) in ans else 1 for i in range(5)])
                else:
                    #We define 0 as hold and 1 as discard
                    actions = agent.get_action(self.state)
                
                self.take_turn(actions)
                self.state = State(self.hand, self.turn_number)
                self.hand_history.append(self.state)
            hand_type, hand_value = self.hand.get_hand()
            if is_human:
                print(f"Final Hand: {self.hand}")
                print(f"Result: {hand_type}, Payout: {hand_value}")
                response = input("Play Again? (y/n)").lower().strip()
                if response == 'no' or response == 'n':
                    print("Goodbye!")
                    playing = False
            else:
                agent.give_reward(hand_value)
                playing = False
            
    def take_turn(self, actions):
        if len(actions) != 5:
            raise ValueError("You must select 5 cards to hold")
        if self.turn_number >= 4:
            raise ValueError("The game is over, no more actions can be taken")
        
        self.turn_number += 1

        for i, action in enumerate(actions):
            if action == 1:
                self.hand.replace_card(i, self.deck.draw())
        self.hand.sort()
        

class State:
    def __init__(self, hand, turn):
        self.hand = hand
        self.turn = turn
    

if __name__ == "__main__":
    VideoPokerGame().play()
    