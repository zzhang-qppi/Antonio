import numpy as np
from chess import Board, move_generation, is_terminal, result
import pandas as pd
import os


class Node:
    def __init__(self, board, father=None):
        self.state = board  # a BoardType object
        self.wins = 0
        self.father = father
        self.children = []
        self.visit_count = 0
        self.children_is_found = False

    def find_children(self):
        self.state.board.expand()
        for child_state in self.state.board.possible_moves:
            self.children.append(Node(child_state, self))
        self.children = np.array(self.children)
        self.children_is_found = True
        # self.update_score()
        return self.children

    # def update_score(self):
    #     for child in self.children:
    #         self.wins += child.wins


# an implementation of Monte Carlo Tree Search
class MCT:
    def __init__(self, root):
        self.root = Node(root)

    def search(self, iterations):
        for i in range(iterations):
            leaf = MCT.select(self.root)
            simulation_result = MCT.rollout(leaf)
            MCT.backpropagate(leaf, simulation_result)
        return MCT.best_child(self.root)

    @staticmethod
    def select(node):
        while node.children_is_found:
            node = MCT.best_child(node)
        if not is_terminal(node.state):
            for child in node.find_children():
                if child.visit_count == 0:
                    return child
        else:
            return node

    @staticmethod
    def expand(node):
        if not node.children_is_found:
            node.find_children()

    #    @staticmethod
    #    def expand(node, depth):
    #        if depth == 0: return
    #        else:
    #            if not node.children_is_found():
    #                node.find_children()
    #            for child in node.children:
    #                MCT.expand(child, depth-1)

    @staticmethod
    def rollout(node):
        while not is_terminal(node.state):
            node = MCT.rollout_policy(node)
        return result(node.state)

    @staticmethod
    def rollout_policy(node):
        return np.random.choice(node.children)

    @staticmethod
    def backpropagate(node, result):  # result = 1 if win 0 if lose
        if node is None: return
        node.wins += result
        MCT.backpropagate(node.father, not result)

    @staticmethod
    def best_child(node, c_param=1.0):
        choices_weights = [
            (child.wins / child.visit_count) + c_param * np.sqrt((np.log(node.visit_count) / child.visit_count))
            for child in node.children
            ]  # UCT selection
        node.children[np.argmax(choices_weights)].visit_count += 1
        return node.children[np.argmax(choices_weights)]


def find_best_action(b, itrns):
    mct = MCT(b)
    return mct.search(itrns).state


def selfplay(max_games, starting_state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
    all_white_game_triples = []
    all_black_game_triples = []
    for game in range(max_games):
        white_game_pairs = []
        black_game_pairs = []
        game_over = False
        state = Board(starting_state)
        while not game_over:
            action = find_best_action(state, 100)
            if state.turn:
                white_game_pairs.append(np.array([state.to_bitboard(), action.to_bitboard()], dtype=np.int8))
            else:
                black_game_pairs.append(np.array([state.to_bitboard(), action.to_bitboard()], dtype=np.int8))
            state, game_over = action, is_terminal(action)
        if not state.turn:
            white_game_triples = np.concatenate((white_game_pairs, np.ones((len(white_game_pairs), 2, 1))), axis=2,
                                                dtype=np.int8)
            black_game_triples = np.concatenate((black_game_pairs, np.zeros((len(black_game_pairs), 2, 1))), axis=2,
                                                dtype=np.int8)
        else:
            white_game_triples = np.concatenate((white_game_pairs, np.zeros((len(white_game_pairs), 2, 1))), axis=2,
                                                dtype=np.int8)
            black_game_triples = np.concatenate((black_game_pairs, np.ones((len(black_game_pairs), 2, 1))), axis=2,
                                                dtype=np.int8)
        assert (white_game_triples.shape[2] == 783)
        assert (black_game_triples.shape[2] == 783)
        all_white_game_triples.append(white_game_triples)
        all_black_game_triples.append(black_game_triples)
    return np.concatenate(all_white_game_triples, dtype=np.int8), np.concatenate(all_black_game_triples, dtype=np.int8)


def selfplay_and_save(max_games, path="./data",
                      starting_state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
    white, black = selfplay(max_games, starting_state)
    white = pd.DataFrame(white)
    black = pd.DataFrame(black)
    white.to_csv(os.path.join(path, "white_games.csv"))
    black.to_csv(os.path.join(path, "black_games.csv"))
