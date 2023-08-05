import numpy as np
import pandas as pd
import torch
import MCTS
import os
from torch.utils.data import Dataset, DataLoader
import datetime

learning_rate = 1e-3
batch_size = 20
epochs = 10
in_features = 8*8*12+1+4+8+1

class ChessDataset(Dataset):
    def __init__(self, data_dir):
        self.labels = pd.read_csv(os.path.join(data_dir, "labels_file"))
        self.inputs = pd.read_csv(os.path.join(data_dir, "inputs_file"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_data = self.inputs.iloc[idx]
        label = self.labels.iloc[idx]
        return input_data, label

class Antonio:
    def __init__(self, load_path=""):
        self.model = self.load_model(load_path)

    @staticmethod
    def load_model(load_path):
        if load_path != "":
            return Antonio.NeuralNetwork()
        else:
            return torch.load(load_path)

    def save_model(self, save_path):
        torch.save(self.model, save_path)

    def train_loop(self, dataloader, loss_fn, optimizer_fn):
        size = len(dataloader.dataset)
        optimizer = optimizer_fn()
        # Set the model to training mode
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            assert X.shape() == (batch_size, )
            # Compute prediction and loss
            pred = self.model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    def test_loop(self, dataloader, loss_fn):
        # Set the model to evaluation mode
        self.model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():  # no calculation of any gradients during test process
            for X, y in dataloader:
                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def train_model(self, dataloader, lr, loss_fn, optimizer_fn):
        optimizer = optimizer_fn(self.model.parameters(), lr=lr)
        costs = []
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            loss = self.train_loop(dataloader, loss_fn, optimizer)
            costs.append(loss)
        return costs

    def selfplay(self, max_games):
        for game in range(max_games):
            white_game_pairs = []
            black_game_pairs = []
            game_over = False
            side = True
            while not game_over:
                sa = np.array(self.mcts(board))
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

    def save_triples(self):
        pass

    def load_triples(self):
        pass

    class NeuralNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = torch.nn.Flatten()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(in_features, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.network(x)
            return logits


lo_fn = torch.nn.MSELoss()
opt_fn = torch.optim.SGD

train_dataloader = DataLoader(ChessDataset("/"), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(ChessDataset("/"), batch_size=batch_size, shuffle=True)

Antonio_1 = Antonio("")

costs = Antonio_1.train_model(train_dataloader, learning_rate, lo_fn, opt_fn)

Antonio_1.save_model('model.pth')