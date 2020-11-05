# agent
import torch
from copy import deepcopy
from collections import namedtuple
named_ex = namedtuple('named_ex', ['state', 'value', 'policy'])
GAME_LEN = 3
import time

class Agent:
    def __init__(self, num_items, num_utterances, nn, opt, num_samples, epsilon, policy_weight, device):
        self.num_items = num_items
        self.num_utterances = num_utterances
        self.nn = nn.to(device)
        self.opt = opt
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.policy_weight = policy_weight
        self.device = device
        self.buffers = tuple(named_ex([], [], []) for _ in range(GAME_LEN))
        self.init_action_repr()
        self.null_action_dynamics = torch.zeros((num_samples, num_utterances))

    def act(self, s, train):
        self.nn.eval()
        _, policies = self.nn(s.tensor().to(self.device))
        policy = policies[s.time()]
        samples = self.one_hot(self.sample(policy), s.time()).to(self.device)
        if s.time() in (0, 1):
            action_dynamics = self.action_dynamics(s, samples)
            next_dists = self.next_distributions(s, samples)
            x = self.make_tensor(s, next_dists)
            with torch.no_grad():
                action_vals, _ = self.nn(x)
            vals = self.average_over_actions(action_vals, action_dynamics)
        else: # final time step
            p1_samples, p2_samples = self.reshape_samples(samples)
            vals = self.compute_terminal_values(s, p1_samples, p2_samples)
            action_dynamics = self.null_action_dynamics
        idx = torch.multinomial(torch.isclose(vals, vals.max()).float(), 1).item()
        if train:
            self.add_to_buffer(s, vals[idx], samples[idx])
            if (torch.rand(1) < self.epsilon): # pick random sample for exploration
                idx = torch.multinomial(torch.ones(self.num_samples), 1).item()
        return samples[idx].to("cpu"), action_dynamics[idx].to("cpu"), vals[idx].item(), (s.time() == 2)

    def add_to_buffer(self, state, value, policy):
        self.buffers[state.time()].state.append(state.tensor().to("cpu"))
        self.buffers[state.time()].value.append(value.to("cpu"))
        self.buffers[state.time()].policy.append(policy.to("cpu"))

    def init_action_repr(self):
        eye = torch.eye(self.num_utterances)
        ls = [eye[i * torch.ones(self.num_samples).long()] for i in range(self.num_utterances)]
        actions = torch.stack(ls, dim=1)
        zeros = torch.zeros((self.num_samples, self.num_utterances, self.num_utterances))
        self.spub1 = torch.cat([actions, zeros], dim=-1).to(self.device) 
        spub2 = []
        for i in range(self.num_utterances):
            tmp = zeros.clone()
            tmp[:, :, i] = 1
            spub2.append(torch.cat([tmp, actions], dim=-1).to(self.device))
        self.spub2 = tuple(spub2)

    def train(self):
        self.opt.zero_grad()
        self.nn.train()
        mse = torch.nn.MSELoss()
        ce = torch.nn.BCELoss()
        loss = 0
        for t in range(GAME_LEN):
            x, v, p = self.get_batch(t)
            v_, policies_ = self.nn(x)
            v_ = v_.view(-1)
            p_ = policies_[t]
            loss += mse(v_, v) + ce(p_, p) * self.policy_weight
        loss.backward()
        self.opt.step()
        self.buffers = tuple(named_ex([], [], []) for _ in range(GAME_LEN))

    def get_batch(self, t):
        x = torch.stack(self.buffers[t].state).to(self.device)
        v = torch.stack(self.buffers[t].value).to(self.device)
        p = torch.stack(self.buffers[t].policy).to(self.device)
        return x, v, p

    def sample(self, policy):
        dist_size = policy.shape[-1]
        shape = (self.num_samples, *policy.shape[:-1])
        flat_policy = policy.view(-1, dist_size)
        samples = torch.multinomial(flat_policy, self.num_samples, replacement=True)
        return samples.permute(1, 0).view(shape)

    def one_hot(self, integer_prescription, time):
        shape = integer_prescription.shape[:-1]
        num_spriv = integer_prescription.shape[-1]
        num_actions = self.num_utterances if time in (0, 1) else self.num_items ** 2
        eye = torch.eye(num_actions)
        return eye[integer_prescription].view(*shape, num_spriv, num_actions)

    def action_dynamics(self, state, prescriptions):
        axis = -1 if state.time() == 0 else -2
        return torch.einsum('ijk,j', prescriptions, state.dist.sum(dim=axis).to(self.device))

    def next_distributions(self, state, prescriptions):
        state_dynamics_string = 'ijk,jl->ikjl' if state.time() == 0 else 'ijk,lj->iklj'
        score = torch.einsum(state_dynamics_string, prescriptions, state.dist.to(self.device))
        norm = score.sum(axis=(2, 3))
        norm[norm == 0] = 1
        return score / norm.view(*score.shape[:2], 1, 1) # N x A x Pri x Pri
        
    def make_tensor(self, state, dist):
        a = self.spub1 if state.time() == 0 else self.spub2[state.spub[0]]
        p1 = dist.sum(axis=-1) # N x A x Pri
        p2 = dist.sum(axis=-2)
        return torch.cat([p1, p2, a], dim=-1)
    
    def average_over_actions(self, action_vals, action_dynamics):
        return (action_vals.view(self.num_samples, -1) * action_dynamics).sum(dim=-1)

    def reshape_samples(self, samples):
        samples_ = samples.view(self.num_samples, 2 * self.num_items, self.num_items, self.num_items)
        samples1 = samples_[:, torch.arange(self.num_items), torch.arange(self.num_items)] # num_samples x num_items x num_items
        samples2 = samples_[:, self.num_items + torch.arange(self.num_items), torch.arange(self.num_items)] # num_samples x num_items x num_items
        return samples1, samples2

    def compute_terminal_values(self, state, p1_prescriptions, p2_prescriptions):
        return (p2_prescriptions.permute(0, 2, 1) * p1_prescriptions * state.dist.to(self.device)).sum(dim=(1, 2))