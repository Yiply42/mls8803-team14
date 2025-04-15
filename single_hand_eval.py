from poker_classes import *
from solution_loader import *
from video_poker import State
import argparse
from play_with_dqn_agent import RLAgent

solved_states = load_solved_states()

def construct_hand_from_input(input_str):
    """
    Constructs a Hand object from a command-line style input string.
    Example input: "AH KD 3C TS 9D"
    """
    rank_map = {rank.value[0]: rank for rank in Rank}
    suit_map = {suit.value[0]: suit for suit in Suit}

    card_strs = input_str.strip().split()
    if len(card_strs) != 5:
        print("A hand must contain exactly 5 cards.")
        return None

    cards = []
    for cs in card_strs:
        if len(cs) != 2:
            print(f"Invalid card format: {cs}")
            return None
        rank_char, suit_char = cs[0], cs[1]
        try:
            rank = rank_map[rank_char.upper()]
            suit = suit_map[suit_char.upper()]
        except KeyError:
            print(f"Invalid card: {cs}")
            return None
        cards.append(Card(rank, suit))

    return Hand(cards)

def construct_position():
    turn = 0
    while turn not in {1, 2, 3}:
        turn = int(input("Enter turn number (1-3): "))
    hand_input = None
    while hand_input is None:
        hand_input = input("Enter your hand (e.g., 'AH KD 3C TS 9D'): ")
        hand_input = construct_hand_from_input(hand_input)
    return (hand_input, turn)

def evaluate_position(hand, turn, baseline_model = None, unlearned_model = None):
    print(f"\nTurn {turn}/3")
    print(f"Hand: {hand} | {hand.get_hand()[0]}")
    # Display optimal action
    
    optimal_action, optimal_ev = solved_states[(tuple(hand.cards), turn)]
    optimal_display = optimal_action.copy()
    for i, q in enumerate(optimal_display):
        if q == 1:
            optimal_display[i] = "Draw"
        else:
            optimal_display[i] = "Hold"
    print(f"Optimal Play: {optimal_display}, EV: {optimal_ev}")
    
    if baseline_model:
        agent_action = baseline_model.get_action(State(hand, turn))
        agent_display = agent_action.tolist()
        for i, q in enumerate(agent_display):
            if q == 1:
                agent_display[i] = "Draw"
            else:
                agent_display[i] = "Hold"
        held_cards = []
        for i, specific_action in enumerate(agent_action):
            if specific_action == 0:
                held_cards.append(hand.cards[i])
            _, action_ev = solved_states[(tuple(held_cards), turn)]
        print(f"Baseline's Play: {agent_display}, EV: {action_ev}")
    
    if unlearned_model:
        agent_action = unlearned_model.get_action(State(hand, turn))
        agent_display = agent_action.tolist()
        for i, q in enumerate(agent_display):
            if q == 1:
                agent_display[i] = "Draw"
            else:
                agent_display[i] = "Hold"
        held_cards = []
        for i, specific_action in enumerate(agent_action):
            if specific_action == 0:
                held_cards.append(hand.cards[i])
            _, action_ev = solved_states[(tuple(held_cards), turn)]
        print(f"Unlearned Agent's Play: {agent_display}, EV: {action_ev}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play Video Poker with a trained DQN agent')
    parser.add_argument('--baseline-model', type=str, default='models/2025-04-14_15-55-40_per_ddqn_nstep3_normal/best_model.pth', help='Path to the trained model')
    parser.add_argument('--test-model', type=str, default='models/2025-04-13_13-28-46_per_ddqn_nstep3_decremental/best_model.pth', help='Path to the unlearned model')
    
    args = parser.parse_args()
    
    unlearned_agent =  RLAgent(args.test_model)
    baseline_agent =  RLAgent(args.baseline_model)
    playing = True
    while playing:
        hand, turn = construct_position()
        evaluate_position(hand, turn, baseline_model=baseline_agent, unlearned_model=unlearned_agent)
        response = input("Play Again? (y/n) ").lower().strip()
        if response == 'no' or response == 'n':
            print("Goodbye!")
            playing = False