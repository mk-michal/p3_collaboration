import logging

import torch

from config import Config
from ddpg import DDPGAgent
from maddpg import MADDPG


class MADDPGUnity(MADDPG):
    def __init__(self, cfg: Config, discount_factor=0.95, tau=0.02):
        super(MADDPGUnity, self).__init__(tau=tau, discount_factor=discount_factor)
        self.logger = logging.getLogger(__name__)
        self.maddpg_agent = [
            DDPGAgent(
                in_actor=24,
                hidden_in_actor=cfg.actor_hidden[0],
                hidden_out_actor=cfg.actor_hidden[1],
                out_actor=2,
                in_critic=52,
                hidden_in_critic=cfg.critic_hidden[0],
                hidden_out_critic=cfg.critic_hidden[1],
                lr_actor=cfg.actor_lr,
                lr_critic=cfg.critic_lr
            ),
            DDPGAgent(
                in_actor=24,
                hidden_in_actor=cfg.actor_hidden[0],
                hidden_out_actor=cfg.actor_hidden[1],
                out_actor=2,
                in_critic=52,
                hidden_in_critic=cfg.critic_hidden[0],
                hidden_out_critic=cfg.critic_hidden[1],
                lr_actor=cfg.actor_lr,
                lr_critic=cfg.critic_lr
            )
        ]


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

        states_next = states_next.permute(1, 0, 2)
        target_critic_input = torch.cat((
            states_next.view(-1, states_next.shape[1] * states_next.shape[2]).float(), target_actions
        ), dim=1).to(device)

        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input.float())

        y = rewards[:, agent_number].view(-1, 1) + self.discount_factor * q_next * (
            1 - done[:,agent_number].long().view(-1,1)
        )

        action = torch.reshape(actions_for_env, shape=(-1, 4))

        critic_input = torch.cat((
            states.view(-1,states.shape[1]*states.shape[2]).float(), action
        ), dim=1).to(device)

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
        states = states.permute(1, 0, 2)
        q_input2 = torch.cat((
            states.view(-1, states.shape[1] * states.shape[2]).float(), q_input
        ), dim=1)

        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        if self.iter % 100 == 0:
            self.logger.info(f'Iteration: {self.iter}. agent_{agent_number}/losses, critic loss: {cl},actor_loss {al}')