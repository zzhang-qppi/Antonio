import numpy as np
from chesssss import Board, result
import pandas as pd
import os
import sys

from fentoimage.board import BoardImage


class Node:
    def __init__(self, board: Board, father=None):
        self.state = board  # a BoardType object
        self.wins = 0
        self.father = father
        self.children = []
        self.visit_count = 1
        self.children_is_found = False

    def find_children(self):
        if self.children_is_found: return self.children
        for child_state in self.state.expand():
            self.children.append(Node(child_state, father=self))
        self.children = np.array(self.children)
        self.children_is_found = True
        #self.update_score()
        return self.children

    def update_score(self):
        for child in self.children:
            self.wins += child.wins


# an implementation of Monte Carlo Tree Search
class MCT:
    def __init__(self, root):
        '''
        root: a Board type object
        '''
        self.root = Node(root)

    def search(self, iterations):
        '''
        iterations: the number of searches. More iterations, better results.
        '''
        for i in range(iterations):
            leaf = MCT.select(self.root)
            end, simulation_result = MCT.rollout(leaf)
            if simulation_result == 1: message = 'white wins'
            if simulation_result == -1: message = 'black wins'
            if simulation_result == 0: message = 'draw'
            MCT.backpropagate(leaf, simulation_result)  # backpropagate the simulation result
            print(f"    iteration {i}: {message}. {i} moves are made. result is backpropagated")
            print('        '+end.state.to_fen())
        return MCT.best_child(self.root)

    @staticmethod
    def select(node: Node):
        cur_node = node
        while cur_node.children_is_found:
            cur_node = MCT.best_child(cur_node)
        if cur_node.state.check_terminal():
            return cur_node
        for child in cur_node.find_children():
            if child.visit_count == 0:
                return child
        return MCT.best_child(cur_node)

    # @staticmethod
    # def expand(node):
    #     if not node.children_is_found:
    #         node.find_children()

    @staticmethod
    def expand(node, depth):
        if depth == 0: return
        for child in node.find_children():
            MCT.expand(child, depth-1)

    @staticmethod
    def rollout(node: Node):
        cur_node = node
        while not cur_node.state.check_terminal():
            cur_node = MCT.rollout_policy(cur_node)
        return cur_node, result(cur_node.state)

    @staticmethod
    def rollout_policy(node) -> Node:
        return np.random.choice(node.find_children())

    @staticmethod
    def backpropagate(node, result):
        '''
        result = 1 if white wins
        = -1 if black wins
        = 0 if draw
        '''
        if node is None: return
        if not node.state.turn and result == 1:  # if the last state was white's move, white's win is recorded
            node.wins += result
        elif node.state.turn and result == -1:  # if the last state was black's move, black's win is recorded
            node.wins += -result
        # node.wins += (node.state.turn is not bool(result+1)) # this line of code does the same logical effect
        MCT.backpropagate(node.father, result)

    @staticmethod
    def best_child(node, c_param=1.0) -> Node:
        choices_weights = [
            (child.wins / child.visit_count) + c_param * np.sqrt((np.log(node.visit_count) / child.visit_count))
            for child in node.children
            ]  # UCT selection
        print(choices_weights)
        node.children[np.argmax(choices_weights)].visit_count += 1
        return node.children[np.argmax(choices_weights)]


def find_best_action(b, itrns) -> Board:
    mct = MCT(b)
    return mct.search(itrns).state


def selfplay(max_games, itrns, starting_state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
    all_white_game_triples = []
    all_black_game_triples = []
    for game in range(max_games):
        print("------------------------------------------")
        print(f'game {game} starts.')
        white_game_pairs = []
        black_game_pairs = []
        game_over = False
        state = Board(starting_state)
        step = 1
        while not game_over:
            action = find_best_action(state, itrns)
            print(f'game {game} step {step}; {"white" if state.turn else "black"} {action.to_fen()}')
            renderer = BoardImage(fen=action.to_fen())
            image = renderer.render()
            image.save(f'./img2/game{game}step{step}.jpg')
            step += 1
            if state.turn:
                white_game_pairs.append(np.array([state.to_bitboard(), action.to_bitboard()], dtype=np.float16))
            else:
                black_game_pairs.append(np.array([state.to_bitboard(), action.to_bitboard()], dtype=np.float16))
            state, game_over = action, action.check_terminal()
        print(f"game {game} is over. {'black' if state.turn else 'white'} wins.")
        print("------------------------------------------")
        if not state.turn:
            white_game_triples = np.concatenate((white_game_pairs, np.ones((len(white_game_pairs), 2, 1))), axis=2,
                                                dtype=np.float16)
            black_game_triples = np.concatenate((black_game_pairs, np.zeros((len(black_game_pairs), 2, 1))), axis=2,
                                                dtype=np.float16)
        else:
            white_game_triples = np.concatenate((white_game_pairs, np.zeros((len(white_game_pairs), 2, 1))), axis=2,
                                                dtype=np.float16)
            black_game_triples = np.concatenate((black_game_pairs, np.ones((len(black_game_pairs), 2, 1))), axis=2,
                                                dtype=np.float16)
        assert (white_game_triples.shape[2] == 783)
        assert (black_game_triples.shape[2] == 783)
        all_white_game_triples.append(white_game_triples)
        all_black_game_triples.append(black_game_triples)
    return np.concatenate(all_white_game_triples, dtype=np.float16), np.concatenate(all_black_game_triples, dtype=np.float16)


def selfplay_and_save(max_games, itrns, path="./data",
                      starting_state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
    white, black = selfplay(max_games, itrns, starting_state)
    white = pd.DataFrame(white, dtype=np.float16)
    black = pd.DataFrame(black, dtype=np.float16)
    white.to_csv(os.path.join(path, "white_games.csv"))
    black.to_csv(os.path.join(path, "black_games.csv"))


sys.setrecursionlimit(100000)
selfplay_and_save(1, 2)