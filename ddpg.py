# individual network settings for each actor + critic pair
# see networkforall for details
from buffer import ReplayBuffer
from utilities import hard_update, gumbel_softmax, onehot_from_logits, soft_update
from torch.optim import Adam
import torch
from torch import nn
import numpy as np

# add OU noise for exploration
from utils import OUNoise

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class DDPGAgent:
    def __init__(
        self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic,
        hidden_in_critic, hidden_out_critic, lr_actor=1.0e-3, lr_critic=1.0e-3,
        noise_dist: str = 'normal', checkpoint_path = None
    ) -> None:
        super(DDPGAgent, self).__init__()

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(
            device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor,
                                    actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.noise = OUNoise(out_actor, scale=1.0, noise_dist=noise_dist)
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.actor.load_state_dict(checkpoint[0]['actor_params'])
            self.target_actor.load_state_dict(checkpoint[0]['actor_params'])
            self.critic.load_state_dict(checkpoint[0]['critic_params'])
            self.target_critic.load_state_dict(checkpoint[0]['critic_params'])

    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor(obs) + noise * self.noise.noise()
        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise * self.noise.noise()
        return action

    def update(
        self,
        buffer: ReplayBuffer,
        batchsize: int = 1000,
        tau: float = 0.005,
        discount: float = 0.98
    ):

        states, actions, rewards, states_next, dones = buffer.sample(batchsize=batchsize)

        actions_next = self.target_actor(torch.stack(states_next).float())
        input_target_critic = torch.cat([torch.stack(states_next).float(), actions_next.float()], axis = 1)
        state_value = self.target_critic(input_target_critic)
        state_value.add_(torch.tensor(rewards).unsqueeze(1))
        state_value = state_value * discount * (1 - torch.tensor(dones).float())
        state_value.detach()

        input_critic = torch.cat([torch.stack(states).float(), torch.stack(actions).float()], axis=1)
        state_value_local = self.critic(input_critic)

        critic_loss = (state_value -  state_value_local).pow(2).mul(0.5).sum(-1).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        actions_new = self.actor(torch.stack(states).float())
        value_critic = self.critic(torch.cat([torch.stack(states).float(), actions_new], axis=1))
        loss_actor = -value_critic.mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        soft_update(self.target_actor, self.actor, tau)
        soft_update(self.target_critic, self.critic, tau)

    def update_targets(self, tau=0.005):
        soft_update(self.target_actor, self.actor, tau)
        soft_update(self.target_critic, self.critic, tau)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)




class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False):
        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.fc1 = nn.Linear(input_dim, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, output_dim)
        self.actor = actor
        # self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        if self.actor:
            # return a vector of the force
            h1 = torch.nn.ReLU()(self.fc1(x))

            h2 = torch.nn.ReLU()(self.fc2(h1))
            h3 = (self.fc3(h2))
            norm = torch.norm(h3)

            # h3 is a 2D vector (a force that is applied to the agent)
            # we bound the norm of the vector to be between 0 and 10
            # return 10.0 * (torch.nn.functional.tanh(norm)) * h3 / norm if norm > 0 else 10 * h3
            return torch.nn.functional.tanh(h3)

        else:
            # critic network simply outputs a number
            h1 = torch.nn.ReLU()(self.fc1(x))
            h2 = torch.nn.ReLU()(self.fc2(h1))
            h3 = (self.fc3(h2))
            return h3
