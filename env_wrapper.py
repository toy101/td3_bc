import gym


class ObservationNormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, mean, std):
        super().__init__(env)
        self.mean = mean
        self.std = std

    def observation(self, observation):
        return (observation - self.mean) / self.std
