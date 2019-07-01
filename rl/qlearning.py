import time

import torch
import torch.nn as nn
import torch.optim as optim

from rl import DQNDense, DQNDuelingDense
from rl import ReplayBuffer, PriorityReplayBuffer
from rl import GreedyPolicy, EpsilonPolicy

from rl import Statistics


class QLearning:

    def __init__(
            self,
            action_size,
            observation_shape,
            soft=True,  # soft update of the target net
            dueling=True,
            double=True,
            noisy=True,
            priority=True,
            ref_net=None,
            replay_buffer_size=10000,
            min_replay_buffer_size=1000,
            target_update_freq=10,
            train_freq=1,
            tau=0.001,
            gamma=0.99,
            hidden_units=128,
            batch_size=128,
            learning_rate=0.001,
            epsilon_start=0.5,
            epsilon_end=0.1,
            beta_decay=200,
            epsilon_decay=200):

        print("QLearning agent:")
        self._double = double
        self._gamma = gamma
        print("\tReward discount (gamma): {}".format(self._gamma))

        # Replay buffer
        self._beta_decay = beta_decay
        if priority:
            self._buffer = PriorityReplayBuffer(
                    capacity=replay_buffer_size,
                    observation_shape=observation_shape)
            print("\tPriority replay buffer is used. Beta decay: {}".format(
                self._beta_decay))
        else:
            self._buffer = ReplayBuffer(
                    capacity=replay_buffer_size,
                    observation_shape=observation_shape)
            print("\tBasic replay buffer is used. Beta parameter is ignored.")
        print("\tReplay buffer size: {}".format(replay_buffer_size))
        self._min_replay_buffer_size = min_replay_buffer_size
        print("\tAmount of records before training starts: {}".format(
                self._min_replay_buffer_size))

        # Target net update
        self._soft = soft
        if self._soft:
            self._tau = tau
            print(("\tTarget net will be soft-updated with τ={}. " +
                   "Target_update_frequency parameter is ignored." +
                   "").format(self._tau))
        else:
            self._target_update_freq = target_update_freq
            print(("\tTarget net will be updated every {} step. " +
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

        if ref_net is not None:
            self._ref_net = ref_net

        # Policies
        self._greedy_policy = GreedyPolicy()
        self._policy = self._greedy_policy
        if noisy:
            print("\tNoisyNet is used. Epsilon parameters are ignored.")
        else:
            print("\tEpsilon. Start: {}; End: {}; Decay: {}".format(
                epsilon_start, epsilon_end, epsilon_decay))
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
        print("\tLearning rate: {}".format(learning_rate))
        self._batch_size = batch_size
        print("\tBatch size: {}".format(self._batch_size))
        self._train_freq = train_freq
        if self._train_freq != 1:
            print("\tTraining {}will be done every {} step".format(
                "(including soft-updates of target network) " if soft else "",
                self._train_freq))

        # Variables which change during training
        self._optimization_step = 0
        self._step = 0
        self.eval = False

    def save(self):
        return {
            "policy_net": self._policy_net,
            "target_net": self._target_net,
        }

    def load(self, props):
        self._policy_net = props["policy_net"]
        self._target_net = props["target_net"]

    def transitions(self, states, actions, rewards, next_states, dones):
        stats = Statistics()
        assert not self.eval
        for idx in range(len(states)):
            self._buffer.push(
                    state=states[idx],
                    action=actions[idx],
                    reward=rewards[idx],
                    next_state=next_states[idx],
                    done=dones[idx])
        stats.set("replay_buffer_size", len(self._buffer))
        if len(self._buffer) >= self._min_replay_buffer_size:
            t0 = time.time()  # time spent for optimization
            stats.set_all(self._optimize())
            stats.set("optimization_time", time.time() - t0)
        return stats

    def step(self, states):
        stats = Statistics()
        self._step += 1

        if not self.eval:
            self._sample_noise()
        states_tensor = torch.from_numpy(states).float().to(self._device)
        self._policy_net.train(False)
        with torch.no_grad():
            q_values = self._policy_net(states_tensor)
        policy = self._greedy_policy if self.eval else self._policy
        actions = policy.get_action(q_values.cpu().numpy())

        if not self.eval:  # During training
            # Do logging
            q = torch.max(q_values).detach()
            stats.set('q', q)
            self._policy_net.log_scalars(stats.set)

            try:
                stats.set('epsilon', self._policy.get_epsilon())
            except AttributeError:
                pass

        return actions

    def _sample_noise(self):
        # Resample noise only during training
        if self.eval:
            return
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

    def _optimize(self):
        stats = Statistics()
        if self.eval:
            return stats
        if self._step % self._train_freq != 0:
            return stats
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
        states, actions, rewards, next_states, term, ids = self._buffer.sample(
                self._batch_size)

        # Make Replay Buffer values consumable by PyTorch
        states = torch.from_numpy(states).float().to(self._device)
        actions = torch.from_numpy(actions).long().to(self._device)
        actions = torch.unsqueeze(actions, dim=1)
        rewards = torch.from_numpy(rewards).float().to(self._device)
        rewards = torch.unsqueeze(rewards, dim=1)
        # For term states the Q value is calculated differently:
        #   Q(term_state) = R
        term_mask = torch.from_numpy(term).to(self._device)
        term_mask = torch.unsqueeze(term_mask, dim=1)
        term_mask = (1 - term_mask).float()
        next_states = torch.from_numpy(next_states).float().to(self._device)

        # Calculate TD Target
        self._sample_noise()
        if self._double:
            # Double DQN: use target_net for Q values estimation of the
            # next_state and policy_net for choosing the action
            # in the next_state.
            next_q_pnet = self._policy_net(next_states).detach()
            next_actions = torch.argmax(next_q_pnet, dim=1).unsqueeze(dim=1)
        else:
            next_q_tnet = self._target_net(next_states).detach()
            next_actions = torch.argmax(next_q_tnet, dim=1).unsqueeze(dim=1)
        self._sample_noise()
        next_q = self._target_net(next_states).gather(
                1, next_actions).detach()  # detach → don't backpropagate

        next_q = next_q * (1 - term_mask).float()  # 0 -> term

        target_q = rewards + self._gamma * next_q

        self._sample_noise()
        q = self._policy_net(states).gather(dim=1, index=actions)

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

        # Debugging of Q-values overestimation
        try:
            true_next_q = self._ref_net(
                    next_states.to('cpu')).detach()
            true_next_actions = torch.argmax(
                    true_next_q, dim=1).unsqueeze(dim=1)
            true_next_v = true_next_q.gather(
                    1, true_next_actions).to(self._device)
            true_next_v = term_mask * true_next_v
            q_overestimate = next_q - true_next_v
            stats.set('q_next_overestimate', q_overestimate.mean().detach())
            stats.set('q_next_err', torch.abs(q_overestimate).mean().detach())
            stats.set('q_next_err_std', torch.std(q_overestimate).detach())
        except AttributeError:
            # No ref_net is set
            pass

        return stats
