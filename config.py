import json
import os
from typing import NamedTuple


class Config(NamedTuple):
    n_episodes = 20000
    buffer_size = int(1e6)
    batchsize = 1000
    noise_beginning = 1.0
    min_noise = 0.1
    tau = 0.005
    warmup = 5000
    actor_hidden = (400,300)
    critic_hidden = (400,300)
    critic_lr = 1e-3
    actor_lr = 5e-4
    update_episode_n = 1
    discount_factor = 0.98
    max_reward = 0.01
    replay_buffer_raward_min = 0.0
    checkpoint_path = 'results/ddpg/episode-10699.pt'
    noise_distribution = 'uniform'


def named_tuple_to_dict(named_tuple: NamedTuple):
    full_dict = {}
    for att in dir(Config):
        if not att.startswith(('_', '__')) and att not in ['count', 'index']:
            full_dict[att] = getattr(named_tuple, att)

    return full_dict

def save_config(file_path: str):
    final_dict = named_tuple_to_dict(Config)
    with open(os.path.join(file_path, 'config_params.json'), 'w') as j:
        json.dump(final_dict, j, indent=4, sort_keys=True)
