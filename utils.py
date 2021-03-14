import numpy as np
import torch

class OUNoise:

    def __init__(self, action_dimension, scale=1, mu=0, theta=0.15, sigma=0.5, noise_dist: str = 'normal'):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.noise_dist = noise_dist
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        if self.noise_dist == 'normal':
            dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(len(x))
        elif self.noise_dist == 'uniform':
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        else:
            raise ValueError('noise_dist needs to be one of [normal, uniform]')
        self.state = x + dx
        return torch.tensor(self.state)