import logging
import random
from collections import deque

import torch

from workspace_udacity.utilities import transpose_list
from workspace_udacity.ddpg import DDPGAgent
from workspace_udacity.maddpg import MADDPG
from unityagents import UnityEnvironment
import numpy as np


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


class MADDPGUnity(MADDPG):
    def __init__(self):
        super(MADDPGUnity, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.maddpg_agent = [
            DDPGAgent(24, 16, 8, 2, 28, 32, 16),
            DDPGAgent(24, 16, 8, 2, 28, 32, 16)]


    def update(self, samples, agent_number, logger,device: str = 'cpu'):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        states, actions_for_env, rewards, states_next, done = list(map(torch.tensor, samples))


        # obs_full = torch.stack(obs_full)
        # next_obs_full = torch.stack(next_obs_full)

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        states_next = states_next.permute(1,0,2)
        target_actions = self.target_act(states_next.float())
        target_actions = torch.cat(target_actions, dim=1)

        target_critic_input = torch.cat((states_next[agent_number].float(), target_actions), dim=1).to(device)

        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input.float())

        y = rewards[:, agent_number].view(-1, 1) + self.discount_factor * q_next * (
            1 - done[:,agent_number].long().view(-1,1)
        )

        action = torch.reshape(actions_for_env, shape=(-1, 4))

        critic_input = torch.cat((states[:, agent_number, :].float(), action), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()

        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        states = states.permute(1,0,2)
        q_input = [self.maddpg_agent[i].actor(ob.float()) if i == agent_number \
                       else self.maddpg_agent[i].actor(ob.float()).detach()
                   for i, ob in enumerate(states)]

        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((states[agent_number].float(), q_input), dim=1)

        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        if self.iter %100 == 0:
            self.logger.info(f'Iteration: {self.iter}. agent_{agent_number}/losses, critic loss: {cl},actor_loss {al}')


def transpose_samples(samples):
    return map(zip())

def main():
    env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64", worker_id=1, seed=1)
    n_episodes = 100000

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    buffer = ReplayBuffer(int(10000))
    maddpg = MADDPGUnity()
    agent0_reward = []
    agent1_reward = []
    batchsize = 1000

    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 2
    noise_reduction = 0.9999
    logger = logging.getLogger('Tennis MADDPG')

    for episode in range(n_episodes):
        reward_this_episode = np.zeros(2)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(2)  # initialize the score (for each agent)
        while True:

            noise *= noise_reduction

            states_tensor = list(map(torch.tensor, states))
            states_tensor = [a.float() for a in states_tensor]
            actions = maddpg.act(states_tensor, noise=noise)
            actions_array = torch.stack(actions).detach().numpy()
            actions_for_env = np.rollaxis(actions_array, 1)
            actions_for_env = np.clip(actions_for_env, -1, 1)  # all actions between -1 and 1

            env_info = env.step(actions_for_env)[brain_name]  # send all actions to tne environment

            states_next = env_info.vector_observations

            buffer_data = (
                states, actions_for_env, env_info.rewards, states_next, env_info.local_done
            )

            buffer.push(buffer_data)

            reward_this_episode += np.array(env_info.rewards)

            dones = env_info.local_done  # see if episode finished
            scores += env_info.rewards  # update the score (for each agent)
            states = states_next  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break

        agent0_reward.append(reward_this_episode[0])
        agent1_reward.append(reward_this_episode[1])
        if len(buffer) > batchsize:
            for i in range(2):
                samples = buffer.sample(batchsize)
                maddpg.update(samples, i, logger)
            maddpg.update_targets()  # soft update the target network towards the actual networks
        if episode %100 == 0 or episode == n_episodes -1:
            logger.info(f'Average 0 reward of agent0 is {np.mean(agent0_reward)}')
            logger.info(f'Average 1 reward of agent1 is {np.mean(agent1_reward)}')

            agent0_reward = []
            agent1_reward = []


if __name__ == '__main__':
    main()