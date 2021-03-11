from typing import NamedTuple


class Config(NamedTuple):
    n_episodes = 20000
    buffer_size = 100000
    batchsize = 1000
    noise = 2
    noise_reduction = 0.99999
    tau = 0.2
    update_episode_n = 1