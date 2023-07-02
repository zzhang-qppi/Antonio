import numpy as np
import tensorflow as tf
from chess import Board, move_generation, is_terminal, result
class Node:
    def __init__(self, board, father = None):
        self.state = board  # a BoardType object
        self.wins = 0
        self.father = father
        self.children = []
        self.visit_count = 0

    def find_children(self):
        for child_state in move_generation(self.state.board):
            self.children.append(Node(child_state, self))
        self.children = np.array(self.children)
        # self.update_score()
        return self.children

    # def update_score(self):
    #     for child in self.children:
    #         self.wins += child.wins

    def children_is_found(self):
        return len(self.children) != 0


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
        while node.children_is_found():
            node = MCT.best_child(node)
        if not is_terminal(node.state):
            for child in node.find_children():
                if child.visit_count == 0:
                    return child
        else:
            return node

    @staticmethod
    def expand(node):
        if not node.children_is_found():
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
    def best_child(node, c_param = 1.0):
        choices_weights = [(child.wins / child.visit_count) + c_param * np.sqrt((np.log(node.visit_count) / child.visit_count))
                           for child in node.children
                           ]  # UCT selection
        return node.children[np.argmax(choices_weights)]

