# posterior
import torch

class Game:
    def __init__(self, num_items, num_utterances):
        self.init_dist = torch.ones((num_items, num_items)) / (num_items ** 2)
        self.num_utterances = num_utterances
        self.min_reward = 0
        self.max_reward = 1

    def init_state(self):
        return State(self.init_dist, self.num_utterances, [])

class State:
    def __init__(self, dist, num_utterances, spub):
        self.dist = dist.clone()
        self.num_utterances = num_utterances
        self.spub = list(spub)

    def next_distribution(self, prescription, action):
        shape = (-1, 1) if self.time() == 0 else (1, -1)
        dist = self.dist * prescription[:, action].view(*shape)
        norm = dist.sum()
        if norm > 0:
            dist = dist / norm
        return dist

    def apply_action(self, prescription, action):
        self.dist = self.next_distribution(prescription, action)
        self.spub.append(action)

    def tensor(self):
        p1 = self.dist.sum(axis=-1)
        p2 = self.dist.sum(axis=-2)
        a = torch.zeros(2 * self.num_utterances)
        if len(self.spub) > 0:
            a[self.spub[0]] = 1
        if len(self.spub) > 1:
            a[self.num_utterances + self.spub[1]] = 1
        return torch.cat([p1, p2, a])

    def clone(self):
        return State(self.dist.clone(), self.num_utterances, self.spub)

    def time(self):
        return len(self.spub)
