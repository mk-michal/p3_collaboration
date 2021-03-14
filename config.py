import json
import os
from typing import NamedTuple


class Config(NamedTuple):
    n_episodes = 10000
    buffer_size = int(1e6)
    batchsize = 100
    noise_beginning = 1
    min_noise = 0.1
    tau = 0.005
    warmup = 4000
    actor_hidden = (200,120)
    critic_hidden = (200,50)
    critic_lr = 1e-5
    actor_lr = 1e-5
    update_episode_n = 1
    discount_factor = 0.98
    max_reward = 0.02
    replay_buffer_raward_min = 0.1
    checkpoint_path = 'data/2021-03-11 16:56:43.095820/episode-4999.pt'
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
