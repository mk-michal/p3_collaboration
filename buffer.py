import random
from collections import deque


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self, transition):
        """push into the buffer"""

        self.deque.append(transition)

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        states = [s[0] for s in samples]
        actions = [s[1] for s in samples]
        rewards = [s[2] for s in samples]
        states_next = [s[3] for s in samples]
        dones = [s[4] for s in samples]
        return states, actions, rewards, states_next, dones

    def __len__(self):
        return len(self.deque)