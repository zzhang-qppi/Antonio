import tensorflow as tf
import numpy as np
import torch

class Antonio():
    def __init__(self, model_file):
        self.net = open()
        self.load
        Antonio.load_model()

    def load_model(self):
        pass

    def selfplay(self, max_games):
        for game in range(max_games):
            white_game_pairs = []
            black_game_pairs = []
            game_over = False
            side = True
            while not game_over:
                sa = np.array(self.mcts(board, eval(net)))
                if side:
                    white_game_pairs.append(sa)
                else:
                    black_game_pairs.append(sa)
                board, game_over = self.next_move(sa[0], sa[1])
                side = not side

            if len(white_game_pairs) > len(black_game_pairs):
                white_game_triples = np.stack((white_game_pairs, [1]*len(white_game_pairs)))
                black_game_triples = np.stack((black_game_pairs, [0] * len(black_game_pairs)))
            elif len(white_game_pairs) == len(black_game_pairs):
                white_game_triples = np.stack((white_game_pairs, [0]*len(white_game_pairs)))
                black_game_triples = np.stack((black_game_pairs, [1] * len(black_game_pairs)))
            else:
                print("numbers of rounds don't match up")

            self.train(white_game_triples)
            self.train(black_game_triples)
        Antonio.save_triples()

    def next_move(self):
        next_board = []
        game_over = False
        return next_board, game_over

    def train(self, game_triples):
        pass

    def save_triples(self):
        pass

    def load_triples(self):
        pass