import logging
from typing import Optional

import torch

from config import Config
from ddpg import DDPGAgent

def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class MADDPGUnity:
    def __init__(
        self, cfg: Config, discount_factor=0.95, tau=0.02, checkpoint_path: Optional[str] = None
    ):
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
                lr_critic=cfg.critic_lr,
                noise_dist=cfg.noise_distribution
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
                lr_critic=cfg.critic_lr,
                noise_dist=cfg.noise_distribution
            )
        ]
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            for i, agent in enumerate(self.maddpg_agent):
                agent.actor.load_state_dict(checkpoint[i]['actor_params'])
                agent.target_actor.load_state_dict(checkpoint[i]['actor_params'])
                agent.critic.load_state_dict(checkpoint[i]['critic_params'])
                agent.target_critic.load_state_dict(checkpoint[i]['critic_params'])

                # agent.actor_optimizer.load_state_dict(checkpoint[i]['actor_optim_params'])
                # agent.critic_optimizer.load_state_dict(checkpoint[i]['critic_optim_params'])

        self.tau = tau
        self.discount_factor = discount_factor
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update_targets(self):
        """soft update targets"""
        # self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)

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
            states_next.view(-1, states_next.shape[1] * states_next.shape[2]).float(), target_actions.float()
        ), dim=1).to(device)

        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input.float())

        y = rewards[:, agent_number].view(-1, 1) + self.discount_factor * q_next * (
            1 - done[:,agent_number].long().view(-1,1)
        )

        action = torch.reshape(actions_for_env, shape=(-1, 4))

        critic_input = torch.cat((
            states.view(-1,states.shape[1]*states.shape[2]).float(), action.float()
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
        q_input_agent_0 = self.maddpg_agent[0].actor(states[0].float())
        q_input_agent_1 = self.maddpg_agent[1].actor(states[0].float())
        if agent_number == 0:
            q_input_agent_0 = q_input_agent_0.detach()
        else:
            q_input_agent_1 = q_input_agent_1.detach()
        # q_input = [self.maddpg_agent[i].actor(ob.float()) if i == agent_number \
        #                else self.maddpg_agent[i].actor(ob.float()).detach()
        #            for i, ob in enumerate(states)]

        q_input = torch.cat([q_input_agent_0, q_input_agent_1], dim=1)

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