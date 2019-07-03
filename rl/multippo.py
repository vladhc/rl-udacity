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
        print("\tNumber of sub-agents: {}".format(n_agents))
        self._n_envs = n_envs
        print("\tNumber of environments: {}".format(self._n_envs))
        assert len(observation_shape) == 1, observation_shape
        self._agents = [
            PPO(
                action_space,
                (observation_shape[0] * n_agents,),
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
        self._action_space = action_space
        self._observation_shape = observation_shape

    def save(self):
        return {
            "agent-{}".format(idx): agent.save()
            for idx, agent in enumerate(self._agents)
        }

    def load(self, props):
        for idx, agent in enumerate(self._agents):
            agent.load(props["agent-{}".format(idx)])

    def step(self, states):
        batch_size = len(states)

        states = self._unite_states(states)
        actions_shape = (batch_size,) + self._action_space.shape
        actions = np.empty(shape=actions_shape, dtype=self._action_space.dtype)
        for agent_idx, agent in enumerate(self._agents):
            agent_actions = agent.step(states)
            actions[agent_idx::self._n_agents, :] = agent_actions

        assert actions.shape == actions_shape, actions.shape
        return actions

    def episodes_end(self):
        for agent in self._agents:
            agent.episodes_end()

    def transitions(self, states, actions, rewards, next_states, term):
        states = self._unite_states(states)
        next_states = self._unite_states(next_states)
        batch_size = self._n_agents * self._n_envs
        actions_shape = (batch_size,) + self._action_space.shape
        assert actions.shape == actions_shape, actions.shape
        stats = Statistics()
        n_agents = self._n_agents
        for idx, agent in enumerate(self._agents):
            s = agent.transitions(
                    states,
                    actions[idx::n_agents],
                    rewards[idx::n_agents],
                    next_states,
                    term[idx::n_agents],
            )
            stats.set_all(s)
        return stats

    def _unite_states(self, states):
        batch_size = len(states)
        assert batch_size == self._n_agents * self._n_envs, batch_size

        united_states = []
        env_batch_size = int(batch_size / self._n_envs)
        for env_idx in range(self._n_envs):
            idx_start = env_idx * env_batch_size
            idx_end = idx_start + env_batch_size
            # Combine the states of the agents into the single united state
            env_state = states[idx_start:idx_end, :]
            env_state = env_state.reshape((1, -1))
            united_shape = (1, self._observation_shape[0] * self._n_agents)
            assert env_state.shape == united_shape, env_state.shape
            united_states.append(env_state)
        return np.concatenate(united_states, axis=0)

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
