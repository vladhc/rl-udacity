import time

import torch
import torch.nn as nn
import torch.optim as optim

from rl import DQNDense, DQNDuelingDense
from rl import ReplayBuffer, PriorityReplayBuffer
from rl import GreedyPolicy, EpsilonPolicy


class QLearning:

    def __init__(
            self,
            sess,
            action_size,
            observation_shape,
            soft=True,  # soft update of the target net
            dueling=True,
            double=True,
            noisy=True,
            priority=True,
            replay_buffer_size=10000,
            target_update_freq=10,
            tau=0.001,
            gamma=0.99,
            hidden_units=128,
            batch_size=128,
            learning_rate=0.001,
            epsilon_start=0.5,
            epsilon_end=0.1,
            beta_decay=200,
            epsilon_decay=200):

        self._double = double
        self._session_id = sess
        self._batch_size = batch_size
        self._gamma = gamma

        # Replay buffer
        self._beta_decay = beta_decay
        if priority:
            self._buffer = PriorityReplayBuffer(replay_buffer_size)
        else:
            self._buffer = ReplayBuffer(replay_buffer_size)

        # Target net update
        self._soft = soft
        if self._soft:
            self._tau = tau
            print(("Target net will be soft-updated with τ={}. " +
                   "Target_update_frequency parameter is ignored." +
                   "").format(self._tau))
        else:
            self._target_update_freq = target_update_freq
            print(("Target net will be updated every {} step. " +
                   "Tau parameter is ignored." +
                   "").format(self._target_update_freq))

        # Target and Policy networks
        if dueling:
            self._policy_net = DQNDuelingDense(
                    observation_shape,
                    action_size,
                    noisy=noisy,
                    hidden_units=hidden_units)
            self._target_net = DQNDuelingDense(
                    observation_shape,
                    action_size,
                    noisy=noisy,
                    hidden_units=hidden_units)
        else:
            self._policy_net = DQNDense(
                    observation_shape,
                    action_size,
                    noisy=noisy,
                    hidden_units=hidden_units)
            self._target_net = DQNDense(
                    observation_shape,
                    action_size,
                    noisy=noisy,
                    hidden_units=hidden_units)

        self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self._policy_net.to(self._device)
        self._target_net.to(self._device)
        self._target_net.train(False)

        # Policies
        self._greedy_policy = GreedyPolicy()
        self._policy = self._greedy_policy
        if not noisy:
            self._policy = EpsilonPolicy(
                    self._policy,
                    action_size,
                    epsilon_start=epsilon_start,
                    epsilon_end=epsilon_end,
                    epsilon_decay=epsilon_decay)

        # Optimizer and loss
        self._loss_fn = nn.MSELoss(reduce=False)
        self._buffer_loss_fn = nn.L1Loss(reduce=False)

        self._optimizer = optim.Adam(
                self._policy_net.parameters(),
                lr=learning_rate)

        # Variables which change during training
        self._optimization_step = 0
        self.prev_state = None
        self.eval = False

    def save_model(self, filename):
        torch.save(self._policy_net, filename)

    def end_episode(self, reward):
        self._store_transition(
                None,
                reward,
                True)
        self.prev_state = None

    def _store_transition(self, state, reward, done):
        if self.eval:
            return
        if self.prev_state is None:  # beginning of the episode
            return
        if done:
            assert state is None
        self._buffer.push(
                self.prev_state,
                self.prev_action,
                reward,
                state)

    def step(self, state, prev_reward, stats):
        self._store_transition(
                state,
                prev_reward,
                False)
        self.prev_action = self._action(state, stats)
        if len(self._buffer) >= self._batch_size:
            t0 = time.time()  # time spent for optimization
            self._optimize(stats)
            stats.set('optimization_time', time.time() - t0)
        stats.set('replay_buffer_size', len(self._buffer))
        self.prev_state = state

        return self.prev_action

    def _action(self, state, stats):
        if not self.eval:
            self._sample_noise()
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._policy_net.train(False)
        with torch.no_grad():
            q_values = self._policy_net(state_tensor)
        policy = self._greedy_policy if self.eval else self._policy
        action = policy.get_action(q_values)

        if not self.eval:  # During training
            # Do logging
            q = torch.max(q_values).detach()
            if self.prev_state is None:
                stats.set('q_start', q)
            stats.set('q', q)
            self._policy_net.log_scalars(stats.set)

            try:
                stats.set('epsilon', self._policy.get_epsilon())
            except AttributeError:
                pass

        return action

    def _sample_noise(self):
        if self.eval:
            return
        # During training
        # Resample noise
        self._policy_net.sample_noise()
        self._target_net.sample_noise()

    def _update_target_net(self):
        if self._soft:
            for target_param, param in zip(
                    self._target_net.parameters(),
                    self._policy_net.parameters()):
                target_param.data.copy_(
                        self._tau * param.data +
                        (1.0 - self._tau)*target_param.data)
        else:
            if self._optimization_step % self._target_update_freq == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())

    def _optimize(self, stats):
        if self.eval:
            return
        self._policy_net.train(True)

        # Increase ReplayBuffer beta parameter 0.4 → 1.0
        # (These numbers are taken from the Rainbow paper)
        beta0 = 0.4
        beta1 = 1.0
        bonus = min(1.0, self._optimization_step / self._beta_decay)
        beta = beta0 + (beta1 - beta0) * bonus
        try:
            self._buffer.set_beta(beta)
            stats.set('replay_beta', beta)
        except AttributeError:
            # In case it's not a PriorityReplayBuffer
            pass
        states, actions, rewards, next_states, ids = self._buffer.sample(
                self._batch_size)

        # Make Replay Buffer values consumable by PyTorch
        states = torch.stack([
            torch.from_numpy(s).float().to(self._device)
            for s in states
        ])
        actions = torch.stack([
            torch.tensor([a], device=self._device) for a in actions
        ])
        rewards = torch.stack([
            torch.tensor([r], dtype=torch.float, device=self._device)
            for r in rewards
        ])
        # For term states the Q value is calculated differently:
        #   Q(term_state) = R
        non_term_mask = torch.tensor(
                [s is not None for s in next_states],
                dtype=torch.uint8, device=self._device)
        non_term_next_states = [
            next_state
            for next_state in next_states
            if next_state is not None
        ]
        non_term_next_states = torch.stack([
            torch.from_numpy(next_state).float().to(self._device)
            for next_state in non_term_next_states
        ])

        # Calculate TD Target
        self._sample_noise()
        if self._double:
            # Double DQN: use target_net for Q values estimation of the
            # next_state and policy_net for choosing the action
            # in the next_state.
            next_q_pnet = self._policy_net(non_term_next_states).detach()
            next_actions = torch.argmax(next_q_pnet, dim=1).unsqueeze(dim=1)
        else:
            next_q_tnet = self._target_net(non_term_next_states).detach()
            next_actions = torch.argmax(next_q_tnet, dim=1).unsqueeze(dim=1)
        self._sample_noise()
        next_q = torch.zeros((self._batch_size, 1), device=self._device)
        next_q[non_term_mask] = self._target_net(non_term_next_states).gather(
                1, next_actions).detach()  # detach → don't backpropagate
        target_q = rewards + self._gamma * next_q

        self._sample_noise()
        q = self._policy_net(states).gather(1, actions)

        loss = self._loss_fn(q, target_q)
        try:
            w = self._buffer.importance_sampling_weights(ids)
            w = torch.from_numpy(w).float().to(self._device)
            loss = w * loss
        except AttributeError:
            # Not a priority replay buffer
            pass
        loss = torch.mean(loss)

        stats.set('loss', loss.detach())

        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

        self._update_target_net()

        self._optimization_step += 1

        # Update priorities in the Replay Buffer
        with torch.no_grad():
            buffer_loss = self._buffer_loss_fn(q, target_q)
            buffer_loss = torch.squeeze(buffer_loss)
            buffer_loss = buffer_loss.cpu().numpy()
            try:
                self._buffer.update_priorities(ids, buffer_loss)
            except AttributeError:
                # That's not a priority replay buffer
                pass
