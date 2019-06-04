import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, namedtuple

from rl import Statistics


class PPO:

    def __init__(
            self,
            action_size,
            observation_shape,
            n_agents,
            gamma=0.99,
            horizon=128,
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

        self._actor_net = Net(
                observation_shape,
                action_size)
        self._actor_net.to(self._device)

        self._critic_net = Net(
                observation_shape,
                1)
        self._critic_net.to(self._device)

        # Optimizer and loss
        self._actor_optimizer = optim.Adam(
                self._actor_net.parameters(),
                lr=learning_rate)
        self._critic_optimizer = optim.Adam(
                self._critic_net.parameters(),
                lr=learning_rate)
        print("\tLearning rate: {}".format(learning_rate))

        self._horizon = horizon
        print("\tHorizon: {}".format(self._horizon))
        self._epochs = epochs
        print("\tEpochs: {}".format(self._epochs))

        self._buffer = TrajectoryBuffer(
                capacity=self._horizon * n_agents,
                horizon=self._horizon,
                gamma=self._gamma)

    def save_model(self, filename):
        torch.save(self._actor_net, filename)

    def step(self, states):
        states_tensor = torch.from_numpy(states).float().to(self._device)
        self._actor_net.train(False)
        with torch.no_grad():
            action_logits = self._actor_net(states_tensor)
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

    def _v(self, state):
        state_tensor = torch.tensor([state]).float().to(self._device)
        assert state_tensor.shape == (1,) + self._observation_shape
        v = self._critic_net(state_tensor).cpu().numpy()
        assert v.shape == (1, 1)
        return v[0][0]

    def _optimize(self):
        stats = Statistics()
        t0 = time.time()
        stats.set('replay_buffer_size', len(self._buffer))
        stats.set('replay_buffer_trajectories',
                len(self._buffer.collected_trajectories))
        if not self._buffer.ready():
            return stats

        self._critic_net.train(True)
        self._actor_net.train(True)

        # Create tensors: state, action, next_state, term
        states, actions, rewards, next_states, term, gs = self._buffer.sample(
                self._v)
        batch_size = len(states)
        assert batch_size == self._buffer.capacity()

        states = torch.from_numpy(states).float().to(self._device)
        actions = torch.from_numpy(actions).long().to(self._device)
        actions = torch.unsqueeze(actions, dim=1)
        rewards = torch.from_numpy(rewards).float().to(self._device)
        rewards = torch.unsqueeze(rewards, dim=1)
        gs = torch.from_numpy(gs).float().to(self._device)
        gs = torch.unsqueeze(gs, dim=1)
        term_mask = torch.from_numpy(term.astype(np.uint8)).to(self._device)
        term_mask = torch.unsqueeze(term_mask, dim=1)
        term_mask = (1 - term_mask).float()  # 0 -> term
        next_states = torch.from_numpy(next_states).float().to(self._device)

        # Estimate advantage
        with torch.no_grad():
            v = self._critic_net(states)
            advantage = gs - v
            advantage = advantage / torch.std(advantage)
            advantage = advantage.detach()
            assert advantage.shape == (batch_size, 1)
            stats.set('advantage', advantage.abs().mean())

        # Critic optimization phase
        loss_fn = nn.MSELoss()
        for _ in range(self._epochs):
            # Critic loss
            v_next = self._critic_net(next_states)
            v_next = v_next * term_mask
            v_next = v_next.detach()
            assert v_next.shape == (batch_size, 1)

            target_v = rewards + self._gamma * v_next
            assert target_v.shape == (batch_size, 1)

            v = self._critic_net(states)
            assert v.shape == (batch_size, 1)

            critic_loss = loss_fn(v, target_v)

            # Critic optimization
            self._critic_optimizer.zero_grad()
            critic_loss.backward()
            self._critic_optimizer.step()

        # Action probabilities of the network before optimization
        old_action_probs = None

        softmax = torch.nn.Softmax(dim=1)
        log_softmax = torch.nn.LogSoftmax(dim=1)

        # Actor optimization phase
        for _ in range(self._epochs):
            action_logits = self._actor_net(states)
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

            # Optimize actor
            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            self._actor_optimizer.step()

            stats.set('loss_actor', actor_loss.detach())
            # Log gradients
            for p in self._actor_net.parameters():
                if p.grad is not None:
                    stats.set('grad_max', p.grad.abs().max().detach())
                    stats.set('grad_mean', (p.grad ** 2).mean().sqrt().detach())

        self._buffer.reset()

        # Log stats
        stats.set('loss_critic', critic_loss.detach())
        stats.set('optimization_time', time.time() - t0)
        stats.set('ppo_optimization_epochs', self._epochs)
        stats.set('ppo_optimization_samples', batch_size)

        # Log entropy metric (opposite to confidence)
        action_logits = self._actor_net(states)
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
            outputs):
        super(Net, self).__init__()

        self.is_dense = len(observation_size) == 1

        hidden_units = 128
        assert self.is_dense

        self.output_head = nn.Sequential(
                nn.Linear(observation_size[0], hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, outputs),
                )

    def forward(self, x):
        return self.output_head(x)


TrajectoryPoint = namedtuple('TrajectoryPoint', [
    'state', 'action', 'reward', 'next_state', 'done'
    ])


class TrajectoryBuffer:

    def __init__(self, capacity, horizon, gamma):
        self._gamma = gamma
        self._capacity = capacity
        self._horizon = horizon
        self.reset()

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

    def sample(self, critic_net):
        self.finish_trajectories()

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        gs = []

        for traj in self.collected_trajectories:
            last_pt = traj[-1]
            if last_pt.done:
                g = 0.0
            else:
                with torch.no_grad():
                    g = critic_net(last_pt.next_state)

            for pt in reversed(traj):
                states.append(pt.state)
                actions.append(pt.action)
                rewards.append(pt.reward)
                next_states.append(pt.next_state)
                dones.append(pt.done)
                g = pt.reward + self._gamma * g
                gs.append(g)

        states = np.asarray(states, dtype=np.float16)
        actions = np.asarray(actions, dtype=np.uint8)
        rewards = np.asarray(rewards, dtype=np.float16)
        next_states = np.asarray(next_states, dtype=np.float16)
        dones = np.asarray(dones, dtype=np.uint8)
        gs = np.asarray(gs, dtype=np.float16)

        return states, actions, rewards, next_states, dones, gs

    def capacity(self):
        return self._capacity
