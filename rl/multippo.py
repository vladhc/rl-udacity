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
            combine_states=False,
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

        # State preprocessor
        state_processor = unite_states if combine_states else noop_states
        self._state_preprocessor, observation_shape = state_processor(
            n_agents,
            n_envs,
            observation_shape,
        )

        # Create agents
        assert len(observation_shape) == 1
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

        states = self._state_preprocessor(states)
        actions_shape = (batch_size,) + self._action_space.shape
        actions = np.empty(shape=actions_shape, dtype=self._action_space.dtype)
        for agent_idx, agent in enumerate(self._agents):
            agent_states = states[agent_idx::self._n_agents]
            agent_actions = agent.step(agent_states)
            actions[agent_idx::self._n_agents, :] = agent_actions

        assert actions.shape == actions_shape, actions.shape
        return actions

    def episodes_end(self):
        for agent in self._agents:
            agent.episodes_end()

    def transitions(self, states, actions, rewards, next_states, term):
        states = self._state_preprocessor(states)
        next_states = self._state_preprocessor(next_states)
        batch_size = self._n_agents * self._n_envs
        actions_shape = (batch_size,) + self._action_space.shape
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


def noop_states(n_agents, n_envs, observation_shape):

    def _noop(states):
        return states

    return _noop, observation_shape


def unite_states(n_agents, n_envs, observation_shape):

    assert len(observation_shape) == 1
    state_size = observation_shape[0] * n_agents

    def _preprocess(states):
        batch_size = len(states)
        assert batch_size == n_agents * n_envs, batch_size

        env_batch_size = int(batch_size / n_envs)

        united_states = []
        for env_idx in range(n_envs):
            idx_start = env_idx * env_batch_size
            idx_end = idx_start + env_batch_size
            # Combine the states of the agents into the single united state
            env_state = states[idx_start:idx_end, :]
            env_state = env_state.reshape((1, -1))
            united_shape = (1, state_size)
            assert env_state.shape == united_shape, env_state.shape
            united_states.append(env_state)
            united_states.append(env_state)

        united_states = np.concatenate(united_states, axis=0)
        assert united_states.shape == (batch_size, state_size)
        return united_states

    return _preprocess, (state_size,)
