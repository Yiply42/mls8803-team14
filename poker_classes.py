from enum import Enum
import random

class Suit(Enum):
    HEART = ("H", 0)
    DIAMOND = ("D", 1)
    CLUB = ("C", 2)
    SPADE = ("S", 3)

    def __str__(self):
        return self.value[0]

class Rank(Enum):
    TWO = ("2", 0)
    THREE = ("3", 1)
    FOUR = ("4", 2)
    FIVE = ("5", 3)
    SIX = ("6", 4)
    SEVEN = ("7", 5)
    EIGHT = ("8", 6)
    NINE = ("9", 7)
    TEN = ("T", 8)
    JACK = ("J", 9)
    QUEEN = ("Q", 10)
    KING = ("K", 11)
    ACE = ("A", 12)

    def __str__(self):
        return self.value[0]
    

class Deck:
    def __init__(self, shuffle = True):
        self.cards = [Card(rank, suit) for suit in Suit for rank in Rank]
        if shuffle:
            self.shuffle()

    def __str__(self):
        str = ""
        for card in self.cards:
            str += f"{card}, "
        return str.strip(", ")
    
    def __len__(self):
        return len(self.cards)
    
    def draw(self):
        return self.cards.pop()
    
    def shuffle(self):
        random.shuffle(self.cards)

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __lt__(self, other):
        if self.rank.value[1] != other.rank.value[1]:
            return self.rank.value[1] < other.rank.value[1]
        return self.suit.value[1] < other.suit.value[1]

    def __gt__(self, other):
        if self.rank.value[1] != other.rank.value[1]:
            return self.rank.value[1] > other.rank.value[1]
        return self.suit.value[1] > other.suit.value[1]

    def __le__(self, other):
        if self.rank.value[1] != other.rank.value[1]:
            return self.rank.value[1] <= other.rank.value[1]
        return self.suit.value[1] <= other.suit.value[1]

    def __ge__(self, other):
        if self.rank.value[1] != other.rank.value[1]:
            return self.rank.value[1] >= other.rank.value[1]
        return self.suit.value[1] >= other.suit.value[1]

    def __ne__(self, other):
        return self.rank != other.rank or self.suit != other.suit
    
    def __repr__(self):
        return str(self)
    
class Hand:
    def __init__(self, cards):
        self.cards = cards
        self.suit_counts = [0] * 4
        self.rank_counts = [0] * 13
        for card in cards:
            self.suit_counts[card.suit.value[1]] += 1
            self.rank_counts[card.rank.value[1]] += 1
        self.sort()

    def __str__(self):
        str = ""
        for card in self.cards:
            str += f"{card}, "
        return str.strip(", ")
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return self.cards == other.cards
    
    def replace_card(self, index_or_card, new_card):
        if isinstance(index_or_card, int):
            self.remove_card(index_or_card)
            self.add_card(new_card, index_or_card)
        else:
            for i in range(len(self.cards)):
                if self.cards[i] == index_or_card:
                    self.remove_card(i)
                    self.add_card(new_card, i)
                    break

    def remove_card(self, index):
        self.rank_counts[self.cards[index].rank.value[1]] -= 1
        self.suit_counts[self.cards[index].suit.value[1]] -= 1
        self.cards[index] = None

    def add_card(self, card, index):
        self.cards[index] = card
        self.rank_counts[card.rank.value[1]] += 1
        self.suit_counts[card.suit.value[1]] += 1

    def sort(self):
        self.cards.sort(reverse = True)

    def get_hand(self):
        """
            returns (hand, hand_value)
        """
        if self.is_royal_flush():
            return ("Royal Flush", 8000)
        if self.is_straight_flush():
            return ("Straight Flush", 500)
        if self.is_four_of_a_kind():
            return ("4-of-a-Kind", 250)
        if self.is_full_house():
            return ("Full House", 90)
        if self.is_flush():
            return ("Flush", 60)
        if self.is_straight():
            return ("Straight", 40)
        if self.is_three_of_a_kind():
            return ("3-of-a-Kind", 30)
        if self.is_two_pair():
            return ("Two Pair", 20)
        if self.is_high_pair():
            return ("High Pair", 10)
        if self.is_pair():
            return ("Low Pair", 5)
        return ("Junk", 0)
    
    def is_royal_flush(self):
        return self.is_straight_flush() and self.cards[0].rank == Rank.TEN
    def is_straight_flush(self):
        return self.is_flush() and self.is_straight()
    def is_four_of_a_kind(self):
        return 4 in self.rank_counts
    def is_full_house(self):
        return 3 in self.rank_counts and 2 in self.rank_counts
    def is_flush(self):
        return 5 in self.suit_counts
    def is_straight(self):
        if self.cards[0].rank == Rank.ACE and self.cards[1].rank == Rank.FIVE and self.cards[2].rank == Rank.FOUR and self.cards[3].rank == Rank.THREE and self.cards[4].rank == Rank.TWO:
            return True

        for i in range(4):
            if self.cards[i].rank.value[1] != self.cards[i + 1].rank.value[1] + 1:
                return False
        return True
    def is_three_of_a_kind(self):
        return 3 in self.rank_counts
    def is_two_pair(self):
        return self.rank_counts.count(2) == 2
    def is_high_pair(self):
        for i in range(Rank.TEN.value[1], Rank.ACE.value[1] + 1):
            if self.rank_counts[i] > 1:
                return True
        return False
    def is_pair(self):
        return 2 in self.rank_counts
    def is_four_to_a_flush(self):
        return 4 in self.suit_counts
    def is_three_to_a_flush(self):
        return 3 in self.suit_counts