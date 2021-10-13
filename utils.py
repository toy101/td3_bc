from pfrl.replay_buffers import ReplayBuffer


def fill_dr4l_pybullet_data(dataset, capacity=None):
    rbuf = ReplayBuffer(capacity)
    dataset_len = dataset["terminals"].size[0]

    for i in range(dataset_len):
        done = bool(dataset["terminals"][i])

        if done:
            next_obs = None
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
