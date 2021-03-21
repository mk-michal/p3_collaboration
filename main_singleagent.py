import datetime
import logging
import os

import matplotlib.pyplot as plt
import torch

from buffer import ReplayBuffer
from ddpg import DDPGAgent
from unityagents import UnityEnvironment
import numpy as np
from config import Config
from config import save_config




def main_single_agent():
    env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64", worker_id=1, seed=1)
    env_date = str(datetime.datetime.now())
    file_path = os.path.join('data_single', env_date)

    os.makedirs(file_path,  exist_ok=True)
    save_config(file_path)

    brain_name = env.brain_names[0]

    buffer = ReplayBuffer(Config.buffer_size)
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

    agent_reward, all_rewards_mean = [], []
    batchsize = Config.batchsize
    max_reward = Config.max_reward
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = Config.noise_beginning

    logger = logging.getLogger('Tennis MADDPG')
    all_rewards = []
    for episode in range(Config.n_episodes):
        reward_this_episode = 0
        env_info = env.reset(train_mode=True)[brain_name]
        states = torch.from_numpy(np.concatenate(env_info.vector_observations))  # get the current state (for each agent)
        scores = np.zeros(2)  # initialize the score (for each agent)
        n_of_steps = 0
        noise = max(Config.min_noise, Config.noise_beginning * (1 - (Config.n_episodes - episode)/Config.n_episodes))
        while True:
            n_of_steps += 1

            states_tensor = torch.tensor(states).float()
            actions = agent.act(states_tensor, noise=noise)
            actions_array = actions.detach().numpy()
            actions_for_env = np.clip(actions_array, -1, 1)  # all actions between -1 and 1

            env_info = env.step(np.array([actions_for_env, actions_for_env]))[brain_name]  # send all actions to tne environment

            states_next = torch.from_numpy(np.concatenate(env_info.vector_observations))

            # if replay_buffer_reward_min is defined, add to replay buffer only the observations higher than min_reward
            reward = np.sum(np.array(env_info.rewards))
            reward_this_episode += reward
            if Config.replay_buffer_raward_min and reward_this_episode >= Config.replay_buffer_raward_min:
                buffer_data = (
                    states, torch.from_numpy(actions_for_env), reward, states_next, env_info.local_done[0]
                )
                buffer.push(buffer_data)

            if not Config.replay_buffer_raward_min:
                buffer_data = (
                    states, torch.from_numpy(actions_for_env), reward, states_next, env_info.local_done[0]
                )

                buffer.push(buffer_data)


            dones = env_info.local_done  # see if episode finished
            scores += np.sum(env_info.rewards)  # update the score (for each agent)
            states = states_next  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break


        all_rewards.append(reward_this_episode)
        all_rewards_mean.append(np.mean(all_rewards[-100:]))
        if len(buffer) > Config.warmup:
            agent.update(buffer, batchsize=batchsize, tau=Config.tau, discount=Config.discount_factor)
            if episode % Config.update_episode_n == 0:
                agent.update_targets(tau=Config.tau)

        if (episode + 1) % 100 == 0 or episode == Config.n_episodes -1:
            logger.info(f'Episode {episode}:  Average reward over 100 episodes is {all_rewards_mean[-1]}')
            if all_rewards_mean and all_rewards_mean[-1] > max_reward:
                logger.info('Found best model. Saving model into file: ...')

                save_dict_list = []
                save_dict = {'actor_params' : agent.actor.state_dict(),
                             'actor_optim_params': agent.actor_optimizer.state_dict(),
                             'critic_params' : agent.critic.state_dict(),
                             'critic_optim_params' : agent.critic_optimizer.state_dict()}

                save_dict_list.append(save_dict)
                save_dict_list.append(save_dict)

                torch.save(save_dict_list, os.path.join(file_path, 'episode-{}.pt'.format(episode)))
                max_reward = all_rewards_mean[-1]
            plt.plot(all_rewards_mean)
            plt.xlabel('N of episodes')
            plt.ylabel('Reward')
            plt.title('Final rewards of single agent for tennis collaboration task')
            plt.savefig(os.path.join(file_path, 'result_plot.png'))

    save_dict = {'actor_params': agent.actor.state_dict(),
                'actor_target_params': agent.target_actor.save_dict(),
                'actor_optim_params': agent.actor_optimizer.state_dict(),
                'critic_params': agent.critic.state_dict(),
                'critic_target_params': agent.target_critic.state_dict(),
                'critic_optim_params': agent.critic_optimizer.state_dict()}


    torch.save(save_dict, os.path.join(file_path, 'episode-{}.pt'.format(episode)))


if __name__ == '__main__':
    main_single_agent()