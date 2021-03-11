import datetime
import logging
import os

import matplotlib.pyplot as plt
import torch

from buffer import ReplayBuffer
from maddpg_tennis import MADDPGUnity
from unityagents import UnityEnvironment
import numpy as np
from config import Config

def main():
    env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64", worker_id=1, seed=1)
    env_date = str(datetime.datetime.now())
    file_path = os.path.join('data', env_date)

    os.makedirs(file_path,  exist_ok=True)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    buffer = ReplayBuffer(Config.buffer_size)
    maddpg = MADDPGUnity()

    agent1_reward, agent0_reward, all_rewards_mean = [], [], []
    batchsize = Config.batchsize
    max_reward = 0
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = Config.noise
    noise_reduction = Config.noise_reduction

    logger = logging.getLogger('Tennis MADDPG')
    all_rewards = []
    for episode in range(Config.n_episodes):
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

        all_rewards.append(max(reward_this_episode[0], reward_this_episode[1]))
        all_rewards_mean.append(np.mean(all_rewards[-100:]))

        agent0_reward.append(reward_this_episode[0])
        agent1_reward.append(reward_this_episode[1])
        if len(buffer) > batchsize:
            for i in range(2):
                samples = buffer.sample(batchsize)
                maddpg.update(samples, i, logger)
            if episode % Config.update_episode_n == 0:
                maddpg.update_targets()  # soft update the target network towards the actual networks


        if (episode + 1) % 100 == 0 or episode == n_episodes -1:
            logger.info(f'Average 0 reward of agent0 is {np.mean(agent0_reward)}')
            logger.info(f'Average 1 reward of agent1 is {np.mean(agent1_reward)}')
            if max(np.mean(agent0_reward), np.mean(agent1_reward)) > max_reward:
                logger.info('Found best model. Saving model into file: ...')

                save_dict_list = []
                for i in range(2):
                    save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                                 'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                                 'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                                 'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}

                    save_dict_list.append(save_dict)
                    save_dict_list.append(save_dict)

                    torch.save(save_dict_list, os.path.join(file_path, 'episode-{}.pt'.format(episode)))
            agent0_reward = []
            agent1_reward = []
            plt.plot(all_rewards_mean)
            plt.savefig(os.path.join(file_path, 'result_plot.png'))




if __name__ == '__main__':
    main()