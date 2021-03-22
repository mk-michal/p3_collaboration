import argparse
import time

import numpy as np
import torch
from unityagents import UnityEnvironment

from config import Config
from ddpg import DDPGAgent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=20, type=int)
    args = parser.parse_args()

    env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64", worker_id=1, seed=1)
    brain_name = env.brain_names[0]


    agent = DDPGAgent(
        in_actor=48,
        hidden_in_actor=Config.actor_hidden[0],
        hidden_out_actor=Config.actor_hidden[1],
        out_actor=2,
        in_critic=50,
        hidden_in_critic=Config.critic_hidden[0],
        hidden_out_critic=Config.critic_hidden[1],
        lr_actor=Config.actor_lr,
        lr_critic=Config.critic_lr,
        noise_dist=Config.noise_distribution,
        checkpoint_path=Config.checkpoint_path

    )


    for episode in range(args.n_episodes):
        env_info = env.reset(train_mode=False)[brain_name]
        states = torch.from_numpy(
        np.concatenate(env_info.vector_observations))  # get the current state (for each agent)
        scores = np.zeros(2)  # initialize the score (for each agent)
        while True:
            states_tensor = torch.tensor(states).float()
            actions = agent.act(states_tensor, noise=0)
            actions_array = actions.detach().numpy()
            actions_for_env = np.clip(actions_array, -1, 1)  # all actions between -1 and 1

            env_info = env.step(np.array([actions_for_env, actions_for_env]))[brain_name]  # send all actions to tne environment

            states_next = torch.from_numpy(np.concatenate(env_info.vector_observations))

            # if replay_buffer_reward_min is defined, add to replay buffer only the observations higher than min_reward
            reward = np.sum(np.array(env_info.rewards))
            dones = env_info.local_done  # see if episode finished
            scores += np.sum(env_info.rewards)  # update the score (for each agent)
            states = states_next  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break

