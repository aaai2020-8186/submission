from agent import Agent
from model import NN
import torch
from torch.optim import Adam
from trainer import TradeComm
from game import Game
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--jobnum", default=0)
args = parser.parse_args()

num_items = 15
num_utterances = 15
input_size = 2 * num_items + 2 * num_utterances
hidden_size = 256
num_samples = 10000
epsilon = 1 / 10
policy_weight = 1 / 5
horizon = 2000
write_every = 10
lr = 1e-4
directory = 'results'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nn = NN(input_size, hidden_size, num_items, num_utterances)
opt = Adam(nn.parameters(), lr=lr)
g = Game(num_items, num_utterances)
agent = Agent(num_items, num_utterances, nn, opt, num_samples, epsilon, policy_weight, device)
trainer = TradeComm(g, agent, directory, args.jobnum)
trainer.run(horizon, write_every)
