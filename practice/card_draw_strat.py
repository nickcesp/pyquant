"""
Poker. 26 red, 26 black. Take one every time, you can choose to guess whether it’s red. You have only one chance.
If you are right, you get 1 dollar. What’s the strategy? And what’s the expected earn?
"""

from random import uniform
import numpy as np


class CardDeck:

    def __init__(self, red_cards=26, black_cards=26):
        self._red_cards = red_cards
        self._black_cards = black_cards

    @property
    def red_cards(self):
        return self._red_cards

    @property
    def black_cards(self):
        return self._black_cards

    @property
    def cards(self):
        return self._red_cards + self._black_cards

    def random_draw(self, put_back=False) -> str:
        """ Draw a card randomly
            :return: Will return red/black
        """
        r_thresh = self._red_cards / (self._red_cards + self._black_cards)
        sample = uniform(0, 1)
        if sample <= r_thresh:
            card = 'red'
            if not put_back: self._red_cards -= 1
        else:
            card = 'black'
            if not put_back: self._black_cards -= 1

        return card


def run_game() -> float:
    """ Simulate a game trial """

    cd = CardDeck()
    profit = 0

    while cd.cards > 0:
        guess = 'red' if cd.red_cards / cd.cards >= 0.5 else 'black'
        if guess == cd.random_draw():
            profit += 1

    return profit


if __name__ == '__main__':

    average_profit = np.average([run_game() for _ in range(10000)])
    print(average_profit)



