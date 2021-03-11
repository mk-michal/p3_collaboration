import json
import os
from typing import NamedTuple


class Config(NamedTuple):
    n_episodes = 10000
    buffer_size = int(1e6)
    batchsize = 1000
    noise = 2
    noise_reduction = 0.9999
    tau = 0.001
    warmup = 2000
    actor_hidden = (200,120)
    critic_hidden = (200,50)
    critic_lr = 1e-3
    actor_lr = 1e-3
    update_episode_n = 1
    discount_factor = 0.95

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
