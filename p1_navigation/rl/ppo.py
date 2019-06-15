import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, namedtuple, deque

from rl import Statistics


class PPO:

    def __init__(
            self,
            action_size,
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
        self._action_size = action_size
        self._gamma = gamma
        print("\tReward discount (gamma): {}".format(self._gamma))
        self._epsilon = epsilon
        print("\tEpsilon: {}".format(self._epsilon))

        self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self._net = Net(
                observation_shape,
                action_size)
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
                horizon=self._horizon,
                gae_lambda=gae_lambda,
                gamma=self._gamma)

    def save_model(self, filename):
        torch.save(self._net, filename)

    def step(self, states):
        states_tensor = torch.from_numpy(states).float().to(self._device)
        self._net.train(False)
        with torch.no_grad():
            action_logits, _ = self._net(states_tensor)
            action_logits = action_logits.double()
            action_probs = torch.nn.Softmax(dim=1)(action_logits)
            action_probs = action_probs.detach()

        actions = []
        for probs in action_probs:
            action = np.random.choice(len(probs), p=probs)
            actions.append(action)

        assert len(actions) == len(states)
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
            _, v = self._net(states_tensor)
            v = v.cpu().numpy()
        assert v.shape == (len(states), 1)
        v = np.squeeze(v, axis=1)
        assert v.shape == (len(states),), v
        return v

    def _optimize(self):
        stats = Statistics()
        t0 = time.time()
        stats.set('replay_buffer_size', len(self._buffer))
        stats.set('replay_buffer_trajectories',
                len(self._buffer.collected_trajectories))
        if not self._buffer.ready():
            return stats

        self._net.train(True)

        # Create tensors: state, action, next_state, term
        states, actions, target_v, advantage = self._buffer.sample(
                self._v)
        batch_size = len(states)
        assert batch_size == self._buffer.capacity()

        states = torch.from_numpy(states).float().to(self._device)
        actions = torch.from_numpy(actions).long().to(self._device)
        actions = torch.unsqueeze(actions, dim=1)
        advantage = torch.from_numpy(advantage).float().to(self._device)
        advantage = torch.unsqueeze(advantage, dim=1)
        target_v = torch.from_numpy(target_v).float().to(self._device)
        target_v = torch.unsqueeze(target_v, dim=1)

        # Estimate advantage
        with torch.no_grad():
            advantage = advantage / torch.std(advantage)
            advantage = advantage.detach()
            assert advantage.shape == (batch_size, 1)
            stats.set('advantage', advantage.abs().mean())

        # Iteratively optimize the network
        softmax = torch.nn.Softmax(dim=1)
        log_softmax = torch.nn.LogSoftmax(dim=1)
        critic_loss_fn = nn.MSELoss()

        # Action probabilities of the network before optimization
        old_action_probs = None

        for _ in range(self._epochs):
            action_logits, v = self._net(states)

            # Calculate Actor Loss
            assert action_logits.shape == (batch_size, self._action_size)

            action_probs = softmax(action_logits).gather(dim=1, index=actions)
            assert action_probs.shape == (batch_size, 1)

            if old_action_probs is None:
                old_action_probs = action_probs.detach()

            r = action_probs / old_action_probs
            assert r.shape == (batch_size, 1)
            obj = torch.min(
                    r * advantage,
                    torch.clamp(
                        r, 1. - self._epsilon, 1. + self._epsilon) * advantage)
            assert obj.shape == (batch_size, 1)

            # minus here because optimizer is going to *minimize* the
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
                    stats.set('grad_mean', (p.grad ** 2).mean().sqrt().detach())

        self._buffer.reset()

        # Log stats
        stats.set('optimization_time', time.time() - t0)
        stats.set('ppo_optimization_epochs', self._epochs)
        stats.set('ppo_optimization_samples', batch_size)

        # Log entropy metric (opposite to confidence)
        action_logits, _ = self._net(states)
        action_probs = softmax(action_logits)
        action_log_probs = log_softmax(action_logits)
        entropy = -(action_probs * action_log_probs).sum(dim=1).mean()
        stats.set('entropy', entropy.detach())

        # Log Kullback-Leibler divergence between the new
        # and the old policy.
        kl = -(
                (action_probs / old_action_probs).log() * old_action_probs
            ).sum(dim=1).mean()
        stats.set('kl', kl.detach())

        return stats


class Net(nn.Module):

    def __init__(
            self,
            observation_size,
            action_size):
        super(Net, self).__init__()

        self.is_dense = len(observation_size) == 1

        hidden_units = 128
        assert self.is_dense

        self.middleware = nn.Sequential(
                nn.Linear(observation_size[0], hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU())

        self.head_actions = nn.Linear(hidden_units, action_size)
        self.head_v = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.middleware(x)
        return self.head_actions(x), self.head_v(x)


TrajectoryPoint = namedtuple('TrajectoryPoint', [
    'state', 'action', 'reward', 'next_state', 'done'
    ])


class TrajectoryBuffer:

    def __init__(self, capacity, horizon, gamma, gae_lambda):
        self._gamma = gamma
        self._capacity = capacity
        self._horizon = horizon
        self.reset()
        self._lambda = gae_lambda
        print("\tlambda for GAE(lambda): {}".format(self._lambda))

    def reset(self):
        self.collected_trajectories = []
        self._trajectories = defaultdict(list)

    def push(self, state, action, reward, next_state, done, env_idx):
        pt = TrajectoryPoint(state, action, reward, next_state, done)
        traj = self._trajectories[env_idx]
        traj.append(pt)
        self._trajectories[env_idx] = traj
        if done or len(traj) == self._horizon:
            self.collected_trajectories.append(traj)
            self._trajectories[env_idx] = []

    def __len__(self):
        return sum([len(traj) for traj in self.collected_trajectories]) + \
                sum([len(traj) for traj in self._trajectories.values()])

    def ready(self):
        return len(self) >= self._capacity

    def finish_trajectories(self):
        for _, traj in self._trajectories.items():
            if len(traj) > 0:
                self.collected_trajectories.append(traj)
        self._trajectories = defaultdict(list)

    def _v(self, traj, v_fn, state_fn):
        states = []
        for pt in traj:
            state = state_fn(pt)
            states.append(state)
        states = np.asarray(states, dtype=np.float16)
        return v_fn(states)

    def sample(self, v_fn):
        self.finish_trajectories()

        states = []
        actions = []
        vs_target = []
        gaes = []  # Generalized Advantage Estimations

        for traj in self.collected_trajectories:

            # States and their values
            vs = self._v(traj, v_fn, lambda pt: pt.state)
            vs_next = self._v(traj, v_fn, lambda pt: pt.next_state)
            if traj[-1].done:
                vs_next[-1] = 0.0

            # We are going to work with the reversed trajectory
            vs_next = np.flip(vs_next)
            vs = np.flip(vs)

            r_trail = []
            v_trail = []

            for idx, pt in enumerate(reversed(traj)):

                v_next = vs_next[idx]
                v = vs[idx]

                vs_target.append(pt.reward + self._gamma * v_next)
                v_trail.append(v_next)
                r_trail.append(pt.reward)

                states.append(pt.state)
                actions.append(pt.action)

                # Calculate the Generalized Advantage Estimation
                assert len(v_trail) == len(r_trail)
                advantages = []
                # how much impact will make each `n_step_advantage`
                # on the resulting `advantage` value.
                weights = []
                for n in range(len(v_trail)):
                    discount = 0

                    # n-step reward
                    n_step_r = 0.
                    for r in list(r_trail)[:n + 1]:
                        n_step_r += r * (self._gamma ** discount)
                        discount += 1

                    # n-step Advantage
                    n_step_advantage = -v + n_step_r + \
                        (self._gamma ** discount) * list(v_trail)[n]
                    advantages.append(n_step_advantage)

                    weights.append(self._lambda ** n)
                gae = 0.0
                for weight, advantage in zip(weights, advantages):
                    gae += weight / sum(weights) * advantage
                gaes.append(gae)

        states = np.asarray(states, dtype=np.float16)
        actions = np.asarray(actions, dtype=np.uint8)
        vs_target = np.asarray(vs_target, dtype=np.float16)
        gaes = np.asarray(gaes, dtype=np.float16)

        assert len(states) == len(actions)
        assert len(states) == len(vs_target)
        assert len(states) == len(gaes)

        return states, actions, vs_target, gaes

    def capacity(self):
        return self._capacity
