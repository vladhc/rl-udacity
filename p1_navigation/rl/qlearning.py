import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

import tensorflow as tf  # for TensorBoard

from rl import DQNDense, DQNDuelingDense
from rl import ReplayBuffer, PriorityReplayBuffer
from rl import GreedyPolicy, EpsilonPolicy


class QLearning:

    def __init__(
            self,
            env,
            session_id,
            dueling=True,
            double=True,
            noisy=True,
            priority=True,
            training_done_fn=lambda x: False,
            replay_buffer_size=10000,
            target_update_freq=10,
            max_episode_steps=200,
            gamma=0.99,
            batch_size=128,
            learning_rate=0.001,
            epsilon_start=0.5,
            epsilon_end=0.1,
            epsilon_decay=200):

        self._env = env
        self._double = double
        self._session_id = session_id
        self._training_done_fn = training_done_fn
        self._max_episode_steps = max_episode_steps
        if priority:
            self._buffer = PriorityReplayBuffer(replay_buffer_size)
        else:
            self._buffer = ReplayBuffer(replay_buffer_size)
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq
        self._gamma = gamma
        self._optimization_step = 0

        self._loss_fn = nn.MSELoss(reduce=False)
        self._buffer_loss_fn = nn.L1Loss(reduce=False)

        try:
            action_size = env.action_space.n
            observation_shape = env.observation_space.shape
        except AttributeError:
            action_size = env.action_size
            observation_shape = (env.state_size, )

        if dueling:
            self._policy_net = DQNDuelingDense(
                    observation_shape,
                    action_size,
                    noisy=noisy)
            self._target_net = DQNDuelingDense(
                    observation_shape,
                    action_size,
                    noisy=noisy)
        else:
            self._policy_net = DQNDense(
                    observation_shape,
                    action_size,
                    noisy=noisy)
            self._target_net = DQNDense(
                    observation_shape,
                    action_size,
                    noisy=noisy)

        self._target_net.train(False)
        self._policy = GreedyPolicy()
        if not noisy:
            self._policy = EpsilonPolicy(
                    self._policy,
                    action_size,
                    epsilon_start=epsilon_start,
                    epsilon_end=epsilon_end,
                    epsilon_decay=epsilon_decay)

        self._optimizer = optim.Adam(
                self._policy_net.parameters(),
                lr=learning_rate)

        summary_file = "./train/{}".format(self._session_id)
        shutil.rmtree(summary_file, ignore_errors=True)
        self._summary_writer = tf.summary.FileWriter(summary_file, None)

    def save_model(self):
        if self._i_episode % 10 != 0:
            return
        filename = 'checkpoins/{}-{}.pth'.format(
                self._session_id,
                self._i_episode)
        torch.save(self._policy_net, filename)

    def train(self, n_episodes):

        for i_episode in range(n_episodes):

            # For logging purposes
            episode_steps = 0
            q_start = None
            q_acc = 0.0     # for calculation of q_avg
            loss_acc = 0.0  # for calculation of loss_avg
            reward_acc = 0.0
            episode_start_time = time.time()
            optimization_time = 0.0  # time spent for optimization
            env_time = 0.0  # time spent for experience generation

            self._i_episode = i_episode
            self.save_model()
            state = self._env.reset()

            for t in range(self._max_episode_steps):
                state_tensor = torch.from_numpy(state).unsqueeze(0)
                self._policy_net.train(False)
                with torch.no_grad():
                    q_values = self._policy_net(state_tensor)

                action = self._policy.get_action(q_values)
                t0 = time.time()
                next_state, reward, done, info = self._env.step(action)
                env_time += (time.time() - t0)
                if done:
                    next_state = None
                self._buffer.push(state, action, reward, next_state)
                state = next_state

                if len(self._buffer) >= self._batch_size:
                    t0 = time.time()
                    loss = self._optimize()
                    optimization_time += time.time() - t0
                    loss_acc += loss

                # For logging
                reward_acc += reward
                q = torch.max(q_values).detach()
                q_acc += q
                if q_start is None:
                    q_start = q

                if done:
                    break
                episode_steps += 1

            episode_end_time = time.time()
            self._log_scalar('episode_steps', episode_steps)
            self._log_scalar('q_start', q_start)
            self._log_scalar('q_avg', q_acc / episode_steps)
            self._log_scalar('reward_avg', reward_acc / episode_steps)
            self._log_scalar('reward', reward_acc)
            self._log_scalar('loss_avg', loss_acc / episode_steps)
            self._log_scalar('replay_buffer_size', len(self._buffer))
            self._log_scalar('time_optimization', optimization_time)
            self._log_scalar('time_exp', env_time)
            self._log_scalar('time_other',
                    episode_end_time - episode_start_time \
                            - optimization_time - env_time)
            self._policy_net.log_scalars(self._log_scalar)
            try:
                self._log_scalar(
                        'epsilon',
                        self._epsilon_greedy_policy.get_epsilon())
            except AttributeError:
                pass
            self._summary_writer.flush()

            if self._training_done_fn(reward_acc):
                self.save_model()
                print("Training completed on {} episode".format(i_episode))
                return


    def _optimize(self):
        self._policy_net.train(True)

        # Increase ReplayBuffer beta parameter 0.4 → 1.0
        # (These numbers are taken from the Rainbow paper)
        beta = 0.4 + (1.0 - 0.4) * (self._i_episode / self._max_episode_steps)
        self._buffer.set_beta(beta)
        states, actions, rewards, next_states, ids = self._buffer.sample(
                self._batch_size)

        # Make Replay Buffer values consumable by PyTorch
        states = torch.stack([torch.from_numpy(s) for s in states])
        actions = torch.stack([
            torch.tensor([a]) for a in actions
        ])
        rewards = torch.stack([
            torch.tensor([r], dtype=torch.float) for r in rewards
        ])
        # For term states the Q value is calculated differently:
        #   Q(term_state) = R
        non_term_mask = torch.tensor(
                [s is not None for s in next_states],
                dtype=torch.uint8)
        non_term_next_states = [
            next_state
            for next_state in next_states
            if next_state is not None
        ]
        non_term_next_states = torch.stack([
            torch.from_numpy(next_state)
            for next_state in non_term_next_states
        ])

        # Calculate TD Target
        if self._double:
            # Double DQN: use target_net for Q values estimation of the
            # next_state and policy_net for choosing the action
            # in the next_state.
            next_q_pnet = self._policy_net(non_term_next_states).detach()
            next_actions = torch.argmax(next_q_pnet, dim=1).unsqueeze(dim=1)
        else:
            next_q_tnet = self._target_net(non_term_next_states).detach()
            next_actions = torch.argmax(next_q_tnet, dim=1).unsqueeze(dim=1)
        next_q = torch.zeros((self._batch_size, 1))
        next_q[non_term_mask] = self._target_net(non_term_next_states).gather(
                1, next_actions).detach()  # detach → don't backpropagate
        target_q = rewards + self._gamma * next_q

        q = self._policy_net(states).gather(1, actions)

        loss = self._loss_fn(q, target_q)
        try:
            w = self._buffer.importance_sampling_weights(ids)
            w = torch.from_numpy(w).float()
            loss = w * loss
        except AttributeError:
            # Not a priority replay buffer
            pass
        loss = torch.mean(loss)

        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

        if self._optimization_step % self._target_update_freq == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())

        self._optimization_step += 1

        # Update priorities in the Replay Buffer
        with torch.no_grad():
            buffer_loss = self._buffer_loss_fn(q, target_q)
            buffer_loss = torch.squeeze(buffer_loss)
            buffer_loss = buffer_loss.numpy()
            try:
                self._buffer.update_priorities(ids, buffer_loss)
            except AttributeError:
                # That's not a priority replay buffer
                pass

        return loss

    def _log_scalar(self, tag, value):
        s = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self._summary_writer.add_summary(s, self._i_episode)
