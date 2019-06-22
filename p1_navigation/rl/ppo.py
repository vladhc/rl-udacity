import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from torch.multiprocessing import Process, Queue, cpu_count

from rl import Statistics
from gym import spaces


class PPO:

    def __init__(
            self,
            action_space,
            observation_shape,
            n_agents,
            gamma=0.99,
            horizon=128,
            gae_lambda=0.95,
            epochs=12,
            epsilon=0.2,
            learning_rate=0.0001):

        print("PPO agent:")

        self._observation_shape = observation_shape
        self._action_space = action_space
        self._gamma = gamma
        print("\tReward discount (gamma): {}".format(self._gamma))
        self._epsilon = epsilon
        print("\tEpsilon: {}".format(self._epsilon))

        self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self._net = Net(observation_shape, action_space)
        self._net.to(self._device)

        # Optimizer and loss
        self._optimizer = optim.Adam(
                self._net.parameters(),
                lr=learning_rate)
        print("\tLearning rate: {}".format(learning_rate))

        self._horizon = horizon
        print("\tHorizon: {}".format(self._horizon))
        self._epochs = epochs
        print("\tEpochs: {}".format(self._epochs))
        self._buffer = TrajectoryBuffer(
                capacity=self._horizon * n_agents,
                traj_constructor=lambda: Trajectory(
                    horizon,
                    observation_shape,
                    self._action_space.dtype,
                    self._action_space.shape),
                horizon=self._horizon,
                gae_lambda=gae_lambda,
                gamma=self._gamma,
                v_fn=self._v)
        if self._is_continous:
            print("\tAction space. Low: {}, high: {}".format(
                self._action_space.low, self._action_space.high))

    @property
    def _is_continous(self):
        return _is_continous(self._action_space)

    def save_model(self, filename):
        torch.save(self._net, filename)

    def step(self, states):
        states_tensor = torch.from_numpy(states).float().to(self._device)
        self._net.train(False)
        batch_size = len(states)

        if self._is_continous:
            action_shape = (batch_size, ) + self._action_space.shape
            with torch.no_grad():
                actions_mu, actions_var, _ = self._net(states_tensor)
                assert actions_mu.shape == action_shape, actions_mu
                assert actions_var.shape == action_shape, actions_var
                actions_arr = []
                for action_idx in range(self._action_space.shape[0]):
                    action_mu = actions_mu[:, action_idx]
                    action_var = actions_var[:, action_idx]
                    assert action_mu.shape == (batch_size,), action_mu.shape
                    assert action_var.shape == (batch_size,), action_var.shape
                    dist = torch.distributions.Normal(
                            action_mu,
                            action_var)
                    sub_actions = dist.sample()
                    actions_arr.append(sub_actions)
                actions = torch.stack(actions_arr, dim=1)
                # Each action can consist of multiple sub-actions
                assert actions.shape == action_shape, actions.shape
        else:
            with torch.no_grad():
                action_logits, _, _ = self._net(states_tensor)
                dist = torch.distributions.categorical.Categorical(
                        logits=action_logits)
                actions = dist.sample()
                assert actions.shape == (len(states),), actions
        actions = actions.detach().cpu().numpy()

        return actions

    def episodes_end(self):
        self._buffer.finish_trajectories()

    def transitions(self, states, actions, rewards, next_states, term):
        assert not self.eval

        for idx in range(len(states)):
            self._buffer.push(
                    states[idx],
                    actions[idx],
                    rewards[idx],
                    next_states[idx],
                    term[idx],
                    idx)
        return self._optimize()

    def _v(self, states):
        assert len(states) > 0
        states_tensor = torch.tensor(states).float().to(self._device)
        assert states_tensor.shape == (len(states),) + self._observation_shape
        with torch.no_grad():
            _, _, v = self._net(states_tensor)
            v = v.cpu().numpy()
        assert v.shape == (len(states), 1)
        v = np.squeeze(v, axis=1)
        assert v.shape == (len(states),), v
        return v

    def _optimize(self):
        stats = Statistics()
        t0 = time.time()
        stats.set('replay_buffer_size', len(self._buffer))
        if not self._buffer.ready():
            return stats

        self._net.train(True)

        # Create tensors: state, action, next_state, term
        states, actions, target_v, advantage = self._buffer.sample()
        batch_size = len(states)
        assert batch_size == self._buffer.capacity()

        states = torch.from_numpy(states).float().to(self._device)
        actions = torch.from_numpy(actions).to(self._device)
        if self._is_continous:
            actions_shape = (batch_size, ) + self._action_space.shape
            actions = actions.float()
        else:
            actions_shape = (batch_size, )
            actions = actions.long()
        assert actions.shape == actions_shape, actions.shape
        target_v = torch.from_numpy(target_v).float().to(self._device)
        target_v = torch.unsqueeze(target_v, dim=1)

        advantage = torch.from_numpy(advantage).float().to(self._device)
        advantage = advantage / advantage.std()
        advantage = advantage.detach()
        assert not torch.isnan(advantage).any(), advantage
        if self._is_continous:
            advantage = torch.unsqueeze(advantage, dim=1)
            assert advantage.shape == (batch_size, 1), advantage.shape
        else:
            assert advantage.shape == (batch_size,)
        stats.set('advantage', advantage.abs().mean())

        # Iteratively optimize the network
        critic_loss_fn = nn.MSELoss()

        # Action probabilities of the network before optimization
        old_log_probs = None
        old_dist = None

        for _ in range(self._epochs):

            # Calculate Actor Loss
            if self._is_continous:
                actions_mu, actions_var, v = self._net(states)
                assert actions_var.shape == actions_shape, actions_var.shape
                assert actions_mu.shape == actions_shape, actions_mu.shape
                assert len(self._action_space.shape) == 1
                log_probs_arr = []
                for action_idx in range(self._action_space.shape[0]):
                    action_mu = actions_mu[:, action_idx]
                    action_var = actions_var[:, action_idx]
                    assert action_mu.shape == (batch_size,), action_mu.shape
                    assert action_var.shape == (batch_size,), action_var.shape
                    dist = torch.distributions.Normal(action_mu, action_var)
                    sub_actions = actions[:, action_idx]
                    assert sub_actions.shape == (batch_size,)
                    log_probs = dist.log_prob(sub_actions)
                    log_probs_arr.append(log_probs)
                log_probs = torch.stack(log_probs_arr, dim=1)
            else:
                action_logits, _, v = self._net(states)
                assert action_logits.shape == (
                        batch_size, self._action_space.n)
                dist = torch.distributions.categorical.Categorical(
                        logits=action_logits)
                log_probs = dist.log_prob(actions)
            assert log_probs.shape == actions_shape, log_probs.shape

            if old_log_probs is None:
                old_log_probs = log_probs.detach()
                old_dist = dist

            r = (log_probs - old_log_probs).exp()

            assert not torch.isnan(r).any(), r
            assert r.shape == actions_shape, r.shape
            obj = torch.min(
                    r * advantage,
                    torch.clamp(
                        r, 1. - self._epsilon, 1. + self._epsilon) * advantage)
            assert obj.shape == actions_shape, obj.shape

            # Minus is here because optimizer is going to *minimize* the
            # loss. If we were going to update the weights manually,
            # (without optimizer) we would remove the -1.
            actor_loss = -obj.mean()

            # Calculate Critic Loss
            assert v.shape == (batch_size, 1)
            assert target_v.shape == (batch_size, 1)

            critic_loss = critic_loss_fn(v, target_v)

            # Optimize
            loss = critic_loss + actor_loss
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            stats.set('loss_actor', actor_loss.detach())
            stats.set('loss_critic', critic_loss.detach())
            # Log gradients
            for p in self._net.parameters():
                if p.grad is not None:
                    stats.set('grad_max', p.grad.abs().max().detach())
                    stats.set(
                            'grad_mean',
                            (p.grad ** 2).mean().sqrt().detach())

        self._buffer.reset()

        # Log stats
        stats.set('optimization_time', time.time() - t0)
        stats.set('ppo_optimization_epochs', self._epochs)
        stats.set('ppo_optimization_samples', batch_size)

        # Log entropy metric (opposite to confidence)
        if self._is_continous:
            action_mu, action_var, _ = self._net(states)
            stats.set('action_variance', action_var.mean().detach())
            stats.set(
                    'action_mu_mean',
                    (action_mu ** 2).mean().sqrt().detach())
            stats.set(
                    'action_mu_max',
                    action_mu.abs().max().detach())

        stats.set('entropy', dist.entropy().mean().detach())

        # Log Kullback-Leibler divergence between the new
        # and the old policy.
        kl = torch.distributions.kl.kl_divergence(dist, old_dist)
        stats.set('kl', kl.mean().detach())

        return stats


class Net(nn.Module):

    def __init__(
            self,
            observation_size,
            action_space):
        super(Net, self).__init__()

        self.is_dense = len(observation_size) == 1
        self._action_space = action_space

        hidden_units = 128
        assert self.is_dense

        self.middleware_critic = nn.Sequential(
                nn.Linear(observation_size[0], hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU())
        self.middleware_actor = nn.Sequential(
                nn.Linear(observation_size[0], hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU())

        self.head_v = nn.Linear(hidden_units, 1)
        if self._is_continous:
            assert len(self._action_space.shape) == 1
            action_size = self._action_space.shape[0]
            self.head_mu = nn.Linear(
                    hidden_units, action_size)
            mu_scale = torch.tensor(
                    (action_space.high - action_space.low) / 2).float()
            mu_offset = torch.tensor(
                    (action_space.high + action_space.low) / 2).float()
            self.register_buffer("mu_scale", mu_scale)
            self.register_buffer("mu_offset", mu_offset)
            print("\tmu scale:", self.mu_scale)
            print("\tmu offset:", self.mu_offset)
            assert self.mu_scale.shape == self._action_space.shape
            assert self.mu_offset.shape == self._action_space.shape
            self.head_variance = nn.Linear(
                    hidden_units, action_size)
        else:
            action_size = action_space.n
            self.head_action_logits = nn.Linear(hidden_units, action_size)

    @property
    def _is_continous(self):
        return _is_continous(self._action_space)

    def forward(self, states):
        x = self.middleware_critic(states)
        v = self.head_v(x)

        x = self.middleware_actor(states)
        assert v.shape == (len(x), 1)
        if self._is_continous:
            mu = self.head_mu(x)
            assert not torch.isnan(mu).any(), mu
            mu = F.tanh(mu)
            mu = mu * self.mu_scale + self.mu_offset
            variance = F.softplus(self.head_variance(x))
            assert not torch.isnan(mu).any(), mu
            assert not torch.isnan(variance).any(), variance
            assert mu.shape == (len(x),) + self._action_space.shape, mu.shape
            assert variance.shape == (len(x),) + self._action_space.shape, \
                variance.shape
            return mu, variance, v
        else:
            action_logits = self.head_action_logits(x)
            return action_logits, None, v


class Trajectory:

    def __init__(self, capacity, observation_shape, action_type, action_shape):
        self._capacity = capacity
        self._cursor = 0

        # +1 here because we store states + one last state
        self._states = np.zeros(
                (capacity + 1,) + observation_shape, dtype=np.float16)
        self.actions = np.zeros((capacity,) + action_shape, dtype=action_type)
        self.rewards = np.zeros(capacity, dtype=np.float16)
        self.terminated = False  # if the final state is term state
        self.gaes = None
        self.v_targets = None

    def push(self, state, action, reward, next_state, done):
        assert not self.terminated
        assert self._cursor < self._capacity

        idx = self._cursor

        self._states[idx] = state
        self._states[idx+1] = next_state
        assert action.shape == self.actions[idx].shape
        self.actions[idx] = action
        self.rewards[idx] = reward

        self._cursor += 1

        if done:
            self.terminated = True
            self.close()

    @property
    def states(self):
        return self._states[:self._cursor]

    @property
    def next_states(self):
        return self._states[1:self._cursor+1]

    @property
    def vs(self):
        return self._vs[:self._cursor]

    @property
    def vs_next(self):
        return self._vs[1:self._cursor+1]

    def update_vs(self, v_fn):
        self._vs = v_fn(self._states)
        if self.terminated:
            self._vs[-1] = 0.0

    # for optimization
    def opimization_cleanup(self):
        self._vs = None
        self.rewards = None

    def close(self):
        # +1 here because we store states + one last state
        self._states = self._states[:self._cursor + 1, :]
        self.actions = self.actions[:self._cursor]
        self.rewards = self.rewards[:self._cursor]

    def done(self):
        return self.terminated or self._capacity == len(self)

    def __len__(self):
        return self._cursor


class TrajectoryBuffer:

    def __init__(
            self,
            capacity,
            traj_constructor,
            horizon,
            gamma,
            gae_lambda,
            v_fn):
        self._gamma = gamma
        self._capacity = capacity
        print("\tlambda for GAE(lambda): {}".format(gae_lambda))
        self._trajectories = defaultdict(traj_constructor)
        self._enrich_queue = Queue()
        self._traj_queue = Queue()
        self._v_fn = v_fn

        enricher_count = cpu_count()
        for _ in range(enricher_count):
            p = Process(
                    target=enrich_trajectories,
                    args=(
                        gamma, horizon, gae_lambda,
                        self._enrich_queue, self._traj_queue))
            p.start()
        for _ in range(enricher_count):
            self._traj_queue.get()

        self.reset()

    def reset(self):
        self._records_collected = 0

    def push(self, state, action, reward, next_state, done, env_idx):
        traj = self._trajectories[env_idx]
        traj.push(state, action, reward, next_state, done)
        self._trajectories[env_idx] = traj
        if traj.done():
            self._enrich_traj(traj)
            del self._trajectories[env_idx]

    def __len__(self):
        return self._records_collected + \
                sum([len(traj) for traj in self._trajectories.values()])

    def _enrich_traj(self, traj):
        self._records_collected += len(traj)
        traj.update_vs(self._v_fn)
        self._enrich_queue.put_nowait(traj)

    def ready(self):
        return len(self) >= self._capacity

    def finish_trajectories(self):
        for _, traj in self._trajectories.items():
            if len(traj) > 0:
                traj.close()
                self._enrich_traj(traj)
        self._trajectories.clear()

    def sample(self):
        self.finish_trajectories()

        v_targets = []
        gaes = []  # Generalized Advantage Estimations

        to_process = self._records_collected

        trajectories = []
        while to_process != 0:
            traj = self._traj_queue.get()
            trajectories.append(traj)
            to_process -= len(traj)

        states = np.concatenate([
            traj.states for traj in trajectories
        ], axis=0)
        actions = np.concatenate([
            traj.actions for traj in trajectories
        ], axis=0)
        v_targets = np.concatenate([
            traj.v_targets for traj in trajectories
        ], axis=0)
        gaes = np.concatenate([
            traj.gaes for traj in trajectories
        ], axis=0)

        assert len(states) == len(actions)
        assert len(states) == len(v_targets)
        assert len(states) == len(gaes)

        return states, actions, v_targets, gaes

    def capacity(self):
        return self._capacity


def enrich_trajectories(
        gamma, horizon, gae_lambda, src_queue, dst_queue):

    dst_queue.put('READY')
    discounts = np.asarray([
        gamma ** n for n in range(horizon + 1)
    ])
    all_weights = np.asarray([
        gae_lambda ** n for n in range(horizon)
    ])

    while True:
        traj = src_queue.get()

        # States and their values
        vs = traj.vs
        vs_next = traj.vs_next

        traj.v_targets = traj.rewards + gamma * vs_next
        assert traj.v_targets.shape == traj.rewards.shape

        traj.gaes = np.zeros(len(traj), dtype=np.float16)
        for idx in range(len(traj)):

            v = vs[idx]

            v_trail = vs_next[idx:]
            r_trail = traj.rewards[idx:]

            # Calculate the Generalized Advantage Estimation
            advantages = np.zeros(len(v_trail), dtype=np.float16)
            for n in range(len(v_trail)):

                # n-step reward
                steps = n + 1
                rewards = r_trail[:steps]
                n_step_r = sum(rewards * discounts[:steps])

                # n-step Advantage
                n_step_advantage = -v + n_step_r + \
                    discounts[steps] * v_trail[n]
                advantages[n] = n_step_advantage

            weights = all_weights[:len(advantages)]
            gae = sum(weights / sum(weights) * advantages)
            traj.gaes[idx] = gae

        traj.opimization_cleanup()

        assert len(traj.states) == len(traj.gaes)
        dst_queue.put_nowait(traj)


def _is_continous(action_space):
    return isinstance(action_space, spaces.Box)
