import numpy as np
import torch
from rl import PPO
from rl import Statistics


class MultiPPO:

    def __init__(
            self,
            action_space,
            observation_shape,
            n_envs,
            n_agents,
            gamma=0.99,
            horizon=128,
            gae_lambda=0.95,
            epochs=12,
            epsilon=0.2,
            learning_rate=0.0001):

        print("MultiPPO agent:")
        self._agents = [
            PPO(
                action_space,
                observation_shape,
                n_envs=n_envs,
                gamma=gamma,
                horizon=horizon,
                gae_lambda=gae_lambda,
                epochs=epochs,
                epsilon=epsilon,
                learning_rate=learning_rate,
            )
            for _ in range(n_agents)
        ]
        self._n_envs = n_envs
        self._action_space = action_space
        self._observation_shape = observation_shape

    def save_model(self, filename):
        multi_model = {}
        for idx, agent in enumerate(self._agents):
            agent_filename = filename + "-agent-{}".format(idx)
            agent.save_model(agent_filename)
            agent_model = torch.load(agent_filename)
            multi_model["agent-{}".format(idx)] = agent_model
        torch.save(multi_model, filename)

    def step(self, states):
        batch_size = len(states)
        assert batch_size == self._n_agents * self._n_envs, batch_size

        actions = []
        agent_batch_size = batch_size / self._n_agents
        for agent_idx in range(self._n_agents):
            agent_states = states[agent_idx::self._n_agents]
            states_shape = (agent_batch_size,) + self._observation_shape
            assert agent_states.shape == states_shape, agent_states.shape

            agent = self._agents[agent_idx]
            agent_actions = agent.step(agent_states)
            actions.append(agent_actions)

        actions = np.concatenate(actions, axis=0)
        actions_shape = (batch_size,) + self._action_space.shape
        assert actions.shape == actions_shape, actions.shape
        return actions

    def episodes_end(self):
        for agent in self._agents:
            agent.episodes_end()

    def transitions(self, states, actions, rewards, next_states, term):
        batch_size = self._n_agents * self._n_envs
        states_shape = (batch_size,) + self._observation_shape
        actions_shape = (batch_size,) + self._action_space.shape
        assert states.shape == states_shape, states.shape
        assert actions.shape == actions_shape, actions.shape
        stats = Statistics()
        n_agents = self._n_agents
        for idx, agent in enumerate(self._agents):
            s = agent.transitions(
                    states[idx::n_agents],
                    actions[idx::n_agents],
                    rewards[idx::n_agents],
                    next_states[idx::n_agents],
                    term[idx::n_agents],
            )
            stats.set_all(s)
        return stats

    @property
    def eval(self):
        return self._eval

    @eval.setter
    def eval(self, v):
        self._eval = v
        for agent in self._agents:
            agent.eval = v

    @property
    def _n_agents(self):
        return len(self._agents)

