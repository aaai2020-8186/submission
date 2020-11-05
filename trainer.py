# coordinator
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import torch

class TradeComm:
    def __init__(self, game, agent, directory, jobnum):
        self.game = game
        self.agent = agent
        self.directory = directory
        self.jobnum = jobnum

    def play_episode(self, train):
        decision_points = [(self.game.init_state(), 1)]
        er = 0
        while len(decision_points) > 0:
            s, prod = decision_points.pop(0)
            prescription, action_dynamics, val, done = self.agent.act(s, train)
            if done:
                er += prod * val
            else:
                for a, p in enumerate(action_dynamics):
                    if p > 0:
                        s_ = s.clone()
                        s_.apply_action(prescription, a)
                        decision_points.append((s_, prod * p))
        return er.item()

    def run(self, horizon, write_every):
        vals = []
        for t in range(horizon):
            self.play_episode(train=True)
            self.agent.train()
            if t % write_every == 0:
                vals.append((t, self.play_episode(train=False)))
                self.write(vals)
    
    def write(self, vals):
        data = {}
        episode_nums, expected_returns = list(zip(*vals))
        data['Episode'] = episode_nums
        data['Expected_Return'] = expected_returns
        data['Jobnum'] = len(vals) * [self.jobnum]
        df = pd.DataFrame(data)
        df.to_pickle('outdir/' + self.directory + '/' + 'job' + str(self.jobnum) + '.pkl')
            