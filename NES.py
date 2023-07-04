import time 
import numpy as np
import torch 
from copy import deepcopy

class Adam(object):
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, truncate=0.0):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        self.t = 0
        self.m = []
        self.v = []

        for p in params:
            self.m.append(torch.zeros_like(p, dtype=torch.float32))
            self.v.append(torch.zeros_like(p, dtype=torch.float32))

        self.truncate = truncate

    def update(self, params, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.b2 ** self.t) / (1.0 - self.b1 ** self.t)
            
        # update the parameters        
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] += (1 - self.b1) * (g - self.m[i])
            self.v[i] += (1 - self.b2) * (g * g - self.v[i])
            
            p.data += lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)
            
            # truncate the weights for the only 4-th layer (6 x 13824)
            if self.truncate > 0.0 and i == 4:
                quantile = np.quantile(p.data.view(-1).detach().numpy(), self.truncate)
                p.data[torch.abs(p.data) < quantile] = 0.0                

class NES_Trainer:
    def __init__(self, agent, learning_rate, noise_std, \
                    noise_decay=1.0, lr_decay=1.0, decay_step=50, norm_rewards=True, truncate=0.0):
        self.agent = agent
        
        self._lr = learning_rate
        self._noise_std = noise_std
        
        self.noise_decay = noise_decay
        self.lr_decay = lr_decay
        self.decay_step = decay_step

        self.norm_rewards = norm_rewards

        self.optimizer = Adam(self.agent.parameters(recurse=True), lr=learning_rate, truncate=truncate)
        self._population = None
        self._count = 0

    @property
    def noise_std(self):
        step_decay = np.power(self.noise_decay, np.floor((1 + self._count) / self.decay_step))

        return self._noise_std * step_decay


    def generate_population(self, npop=50):
        self._population = []

        for i in range(npop):
            new_agent = deepcopy(self.agent)
            new_agent.E = [] 

            for i, layer in enumerate(new_agent.parameters()):
                noise = torch.randn(size=layer.shape, dtype=torch.float32)
                new_agent.E.append(noise)
                layer.data = layer.data + self.noise_std * noise

            self._population.append(new_agent)

        return self._population

    def update_agent(self, rewards, reward_process):
        if self._population is None:
            raise ValueError("populations is none, generate & eval it first")

        if reward_process == 'standardize':
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        elif reward_process == 'centered_rank':
            rewards = self.compute_centered_ranks(rewards)
        
        # compute the gradient for each layer 
        gradients = []
        for i, layer in enumerate(self.agent.parameters()):
            # create a matrix to store the gradient for each agent using torch 
            w_updates = torch.zeros_like(layer, dtype=torch.float32)
            for j, explore_agent in enumerate(self._population):
                w_updates = w_updates + (explore_agent.E[i] * rewards[j])
            w_updates /= len(rewards) * self.noise_std
            gradients.append(w_updates)
        # update the weights by using optimizer
        self.optimizer.update(params=self.agent.parameters(), grads=gradients)
        self._count = self._count + 1

    def get_agent(self):
        return self.agent

    def compute_centered_ranks(self, x): 
        assert x.ndim == 1 
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        ranks = ranks.astype(np.float32)
        ranks /= (x.size - 1)
        ranks -= .5
        return ranks
