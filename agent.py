import collections
import copy
from logging import getLogger

import numpy as np
import torch
from torch.nn import functional as F

import pfrl
from pfrl.agents import TD3
from pfrl.agent import AttributeSavingMixin, BatchAgent
from pfrl.replay_buffer import ReplayUpdater, batch_experiences
from pfrl.utils import clip_l2_grad_norm_
from pfrl.utils.batch_states import batch_states
from pfrl.utils.copy_param import synchronize_parameters


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def default_target_policy_smoothing_func(batch_action):
    """Add noises to actions for target policy smoothing."""
    noise = torch.clamp(0.2 * torch.randn_like(batch_action), -0.5, 0.5)
    return torch.clamp(batch_action + noise, -1, 1)

class DemoReplayUpdater(object):
    """Object that handles update schedule and configurations.
    Args:
        replay_buffer (mod.buffers.PrioritizedDemoReplayBuffer): Bbuffer for self-play
        update_func (callable): Callable that accepts one of these:
            (1) two lists of transition dicts (if episodic_update=False)
            (2) two lists of transition dicts (if episodic_update=True)
        batch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        n_times_update (int): Number of repetition of update
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
    """

    def __init__(self, replay_buffer,
                 update_func, batch_size, episodic_update,
                 n_times_update, replay_start_size, update_interval,
                 episodic_update_len=None):
        assert batch_size <= replay_start_size
        self.replay_buffer = replay_buffer
        self.update_func = update_func
        self.batch_size = batch_size
        self.episodic_update = episodic_update
        self.episodic_update_len = episodic_update_len
        self.n_times_update = n_times_update
        self.replay_start_size = replay_start_size
        self.update_interval = update_interval

    # def update_if_necessary(self, iteration):
    #     """Called during normal self-play
    #     """
    #     if len(self.replay_buffer) < self.replay_start_size:
    #         return
    #
    #     if (self.episodic_update and (
    #             self.replay_buffer.n_episodes < self.batch_size)):
    #         return
    #
    #     if iteration % self.update_interval != 0:
    #         return
    #
    #     for _ in range(self.n_times_update):
    #         if self.episodic_update:
    #             raise NotImplementedError()
    #         else:
    #             transitions_demo = self.replay_buffer.sample(
    #                 self.batch_size)
    #             self.update_func(transitions_demo)
    #             # Update beta only during RL
    #             self.replay_buffer.update_beta()

    def update_from_demonstrations(self):
        """Called during pre-train steps. All samples are from demo buffer
        """
        # if self.episodic_update:
            # episodes_demo = self.replay_buffer.sample_episodes(
            #     self.batch_size, self.episodic_update_len)
            # self.update_func(episodes_demo)
        assert not self.episodic_update, "Not adapt episodic update"

        transitions_demo = self.replay_buffer.sample(self.batch_size)
        self.update_func(transitions_demo)


class TD3PlusBC(TD3):
    """A Minimalist Approach to Offline Reinforcement Learning.
    See https://arxiv.org/abs/2106.06860
    Args:
        policy (Policy): Policy.
        q_func1 (Module): First Q-function that takes state-action pairs as input
            and outputs predicted Q-values.
        q_func2 (Module): Second Q-function that takes state-action pairs as
            input and outputs predicted Q-values.
        policy_optimizer (Optimizer): Optimizer setup with the policy
        q_func1_optimizer (Optimizer): Optimizer setup with the first
            Q-function.
        q_func2_optimizer (Optimizer): Optimizer setup with the second
            Q-function.
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        phi (callable): Feature extractor applied to observations
        soft_update_tau (float): Tau of soft target update.
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
        burnin_action_func (callable or None): If not None, this callable
            object is used to select actions before the model is updated
            one or more times during training.
        policy_update_delay (int): Delay of policy updates. Policy is updated
            once in `policy_update_delay` times of Q-function updates.
        target_policy_smoothing_func (callable): Callable that takes a batch of
            actions as input and outputs a noisy version of it. It is used for
            target policy smoothing when computing target Q-values.

        OfflineRL parameters
        n_train_steps (int)

        TD3+BC specific parameters
        alpha (float): Actor update parameter
        mean, std (float, float): State Normalization
    """

    saved_attributes = (
        "policy",
        "q_func1",
        "q_func2",
        "target_policy",
        "target_q_func1",
        "target_q_func2",
        "policy_optimizer",
        "q_func1_optimizer",
        "q_func2_optimizer",
    )

    def __init__(
            self,
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            replay_buffer,
            gamma,
            explorer,
            gpu=None,
            replay_start_size=10000,
            minibatch_size=100,
            update_interval=1,
            phi=lambda x: x,
            soft_update_tau=5e-3,
            n_times_update=1,
            max_grad_norm=None,
            logger=getLogger(__name__),
            batch_states=batch_states,
            burnin_action_func=None,
            policy_update_delay=2,
            target_policy_smoothing_func=default_target_policy_smoothing_func,
            # OfflineRL parameters
            n_train_steps=1,
            # TD3+BC specific parameters
            alpha=2.5,
            mean=0.0,
            std=1.0,
            log_interval=5000
    ):

        super(TD3PlusBC, self).__init__(policy,
                                        q_func1,
                                        q_func2,
                                        policy_optimizer,
                                        q_func1_optimizer,
                                        q_func2_optimizer,
                                        replay_buffer,
                                        gamma,
                                        explorer,
                                        gpu=None,
                                        replay_start_size=10000,
                                        minibatch_size=100,
                                        update_interval=1,
                                        phi=lambda x: x,
                                        soft_update_tau=5e-3,
                                        n_times_update=1,
                                        max_grad_norm=None,
                                        logger=getLogger(__name__),
                                        batch_states=batch_states,
                                        burnin_action_func=None,
                                        policy_update_delay=2,
                                        target_policy_smoothing_func=default_target_policy_smoothing_func
                                        )
        self.n_train_steps=n_train_steps*len(self.replay_buffer)
        self.policy = policy
        self.q_func1 = q_func1
        self.q_func2 = q_func2

        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.policy.to(self.device)
            self.q_func1.to(self.device)
            self.q_func2.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.phi = phi
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        self.policy_optimizer = policy_optimizer
        self.q_func1_optimizer = q_func1_optimizer
        self.q_func2_optimizer = q_func2_optimizer
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=minibatch_size,
            n_times_update=1,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
            episodic_update=False,
        )
        self.max_grad_norm = max_grad_norm
        self.batch_states = batch_states
        self.burnin_action_func = burnin_action_func
        self.policy_update_delay = policy_update_delay
        self.target_policy_smoothing_func = target_policy_smoothing_func

        self.t = 0
        self.policy_n_updates = 0
        self.q_func_n_updates = 0
        self.last_state = None
        self.last_action = None

        # Target model
        self.target_policy = copy.deepcopy(self.policy).eval().requires_grad_(False)
        self.target_q_func1 = copy.deepcopy(self.q_func1).eval().requires_grad_(False)
        self.target_q_func2 = copy.deepcopy(self.q_func2).eval().requires_grad_(False)

        # TD3+BC specific parameters
        self.alpha = alpha
        self.mean = mean
        self.std = std

        self.replay_updater = DemoReplayUpdater(
            replay_buffer=self.replay_buffer,
            update_func=self.update,
            batch_size=minibatch_size,
            episodic_update=False,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )

        # Statistics
        self.q1_record = collections.deque(maxlen=1000)
        self.q2_record = collections.deque(maxlen=1000)
        self.q_func1_loss_record = collections.deque(maxlen=100)
        self.q_func2_loss_record = collections.deque(maxlen=100)
        self.policy_loss_record = collections.deque(maxlen=100)

        self.tpre = 0
        self.log_interval = log_interval

    def update_policy(self, batch):
        """Compute loss for actor."""

        batch_state = batch["state"]
        batch_action = batch["action"]

        onpolicy_actions = self.policy(batch_state).rsample()
        q = self.q_func1((batch_state, onpolicy_actions))

        # Compute actor loss with Behavioral Cloning
        lmbda = self.alpha / q.abs().mean().detach()
        loss = -lmbda * torch.mean(q) + F.mse_loss(onpolicy_actions, batch_action)

        self.policy_loss_record.append(float(loss))
        self.policy_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        self.policy_n_updates += 1

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""

        batch = batch_experiences(experiences, self.device, self.phi, self.gamma)
        self.update_q_func(batch)
        if self.q_func_n_updates % self.policy_update_delay == 0:
            if self.q_func_n_updates % self.log_interval == 0:
                self.logger.info('offlineRL-step:%s statistics:%s',
                                 self.tpre, self.get_statistics())
            self.update_policy(batch)
            self.sync_target_network()

    def batch_select_onpolicy_action(self, batch_obs):
        with torch.no_grad(), pfrl.utils.evaluating(self.policy):
            batch_xs = self.batch_states(batch_obs, self.device, self.phi)

            # state normalization
            batch_xs = (batch_xs - self.mean) / self.std
            batch_action = self.policy(batch_xs).sample().cpu().numpy()
        return list(batch_action)

    def offline_train(self, epoch):
        """Uses purely expert demonstrations to do pre-training
        """
        for tpre in range(self.n_train_steps * epoch + 1, self.n_train_steps * (epoch + 1) + 1):
            self.tpre = tpre
            self.replay_updater.update_from_demonstrations()
