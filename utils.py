import numpy as np
import torch
from pfrl.replay_buffers import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_states(obses, eps=1e-3):
    obses = torch.FloatTensor(obses).to(device)

    std, mean = obses.std_mean(0, keepdims=True)
    std = std + eps

    obses = (obses - mean) / std
    obses = obses.to('cpu').detach().numpy().copy()

    return obses, mean, std


def fill_dr4l_pybullet_data(dataset, mean=0.0, std=1.0, capacity=None):
    rbuf = ReplayBuffer(capacity)
    dataset_len = dataset["terminals"].size
    obs_size = dataset["observations"][0].size
    terminal_next_obs = (np.random.normal(loc=mean, scale=std,
                                          size=obs_size) - mean) / std  # This array has no meaning.

    for i in range(dataset_len):
        done = bool(dataset["terminals"][i])

        if done:
            next_obs = terminal_next_obs
        else:
            next_obs = dataset["observations"][i + 1]

        rbuf.append(state=dataset["observations"][i],
                    action=dataset["actions"][i],
                    reward=dataset["rewards"][i],
                    next_state=next_obs,
                    next_action=None,
                    is_state_terminal=done)
        if done:
            rbuf.stop_current_episode()

    return rbuf
