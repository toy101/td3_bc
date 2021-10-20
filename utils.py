import numpy as np
import torch
from pfrl.replay_buffers import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_states(obses, eps=1e-3):
    obses = torch.FloatTensor(obses).to(device)

    std, mean = torch.std_mean(obses, dim=0)
    std = std + eps

    obses = (obses - mean) / std
    obses = obses.to('cpu').detach().numpy().copy()

    return obses, mean, std


def fill_dr4l_pybullet_data(dataset, mean, std, capacity=None):

    rbuf = ReplayBuffer(capacity)
    dataset_len = dataset["terminals"].size
    obs_size = dataset["observations"][0].size

    terminal_next_obs = torch.empty(size=(obs_size,))

    for i, (m, s) in enumerate(zip(mean, std)):
        ramdom = torch.normal(mean=m, std=s)
        terminal_next_obs[i] = ramdom
    terminal_next_obs = (terminal_next_obs - mean) / std # This array has no meaning.
    terminal_next_obs = terminal_next_obs.to('cpu').detach().numpy().copy()

    for i in range(dataset_len):
        done = bool(dataset["terminals"][i])

        if done:
            next_obs = terminal_next_obs
        elif i == dataset_len - 1:
            rbuf.stop_current_episode()
            break
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
