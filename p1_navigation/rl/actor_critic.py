import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl import Statistics


class ActorCritic:

    def __init__(
            self,
            action_size,
            observation_shape,
            gamma=0.99,
            learning_rate=0.0001):

        print("Actor-Critic agent:")

        self._observation_shape = observation_shape
        self._gamma = gamma
        print("\tReward discount (gamma): {}".format(self._gamma))

        self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self._net = ActorCriticNet(
                observation_shape,
                action_size)
        self._net.to(self._device)

        # Optimizer and loss
        self._optimizer = optim.Adam(
                self._net.parameters(),
                lr=learning_rate)
        print("\tLearning rate: {}".format(learning_rate))

        # Variables which change during training
        self.prev_state = None
        self._trajectory = []

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
        return actions

    def transitions(self, states, actions, rewards, next_states, term):
        stats = Statistics()
        if self.eval:
            return stats
        t0 = time.time()
        self._net.train(True)

        states = torch.from_numpy(states).float().to(self._device)
        actions = torch.tensor(actions).long().to(self._device)
        actions = torch.unsqueeze(actions, dim=1)
        rewards = torch.from_numpy(rewards).float().to(self._device)
        rewards = torch.unsqueeze(rewards, dim=1)
        term_mask = torch.from_numpy(term.astype(np.uint8)).to(self._device)
        term_mask = torch.unsqueeze(term_mask, dim=1)
        next_states = torch.from_numpy(next_states).float().to(self._device)

        _, v_next = self._net(next_states)
        v_next = v_next * (1 - term_mask).float()  # 0 -> term
        action_logits, v = self._net(states)

        # it's used as:
        # 1. loss for the critic
        # 2. advantage for the actor
        delta = rewards + self._gamma * v_next - v

        critic_loss = delta.abs().mean()

        log_softmax = torch.nn.LogSoftmax(dim=1)
        action_log_probs = log_softmax(action_logits)
        action_log_probs = action_log_probs.gather(dim=1, index=actions)

        # minus here because optimizer is going to *minimize* the
        # loss. If we were going to update the weights manually,
        # (without optimizer) we would remove the -1.
        actor_loss = - (delta * action_log_probs).mean()

        loss = actor_loss + critic_loss

        # Optimize
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        stats.set('loss', loss.detach())
        stats.set('loss_critic', critic_loss.detach())
        stats.set('loss_actor', actor_loss.detach())
        stats.set('optimization_time', time.time() - t0)

        # Log entropy metric (opposite to confidence)
        action_probs = torch.nn.Softmax(dim=1)(action_logits)
        entropy = -(action_probs * action_log_probs).sum(dim=1).mean()
        stats.set('entropy', entropy.detach())

        # Log gradients
        for p in self._net.parameters():
            if p.grad is not None:
                stats.set('grad_max', p.grad.abs().max().detach())
                stats.set('grad_mean', (p.grad ** 2).mean().sqrt().detach())

        # Log Kullback-Leibler divergence between the new
        # and the old policy.
        new_action_logits, _ = self._net(states)
        new_action_probs = torch.nn.Softmax(dim=1)(new_action_logits)
        kl = -(
                (new_action_probs / action_probs).log() * action_probs
            ).sum(dim=1).mean()
        stats.set('kl', kl.detach())
        return stats


class ActorCriticNet(nn.Module):

    def __init__(
            self,
            observation_size,
            action_size):
        super(ActorCriticNet, self).__init__()

        self.is_dense = len(observation_size) == 1

        hidden_units = 128
        assert self.is_dense

        self.base = nn.Sequential(
                nn.Linear(observation_size[0], hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU())

        self.action_head = nn.Linear(hidden_units, action_size)
        self.value_head = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = self.base(x)
        actions = self.action_head(x)
        value = self.value_head(x)
        return actions, value
