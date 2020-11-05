import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_items, num_utterances):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fcv = nn.Linear(hidden_size, 1)
        self.fcp1 = nn.Linear(hidden_size, num_items * num_utterances)
        self.fcp2 = nn.Linear(hidden_size, num_items * num_utterances)
        self.fctrade = nn.Linear(hidden_size, 2 * num_items ** 3)
        self.num_utterances = num_utterances
        self.num_items = num_items
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x) + x)
        x = F.relu(self.fc3(x) + x)
        v = self.fcv(x)
        p1 = self.softmax(self.fcp1(x).view(*batch_shape, self.num_items, self.num_utterances))
        p2 = self.softmax(self.fcp2(x).view(*batch_shape, self.num_items, self.num_utterances))
        trade = self.softmax(self.fctrade(x).view(*batch_shape, 2 * self.num_items, self.num_items ** 2))
        return v, (p1, p2, trade)