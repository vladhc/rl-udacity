import numpy as np
from rl import PPO
from rl import Statistics


class MultiPPO:

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

        print("MultiPPO agent:")
        self._agents = [
            PPO(
                action_space,
                observation_shape,
                n_agents=1,
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
        self._observation_space = observation_shape

    def save_model(self, filename):
        for idx, agent in enumerate(self._agents):
            agent.save_model(filename + "-agent-{}".format(idx))

    def step(self, states):
        actions = []
        for state, agent in zip(states, self._agents):
            state = np.expand_dims(state, axis=0)
            assert state.shape == (1,) + self._observation_space, state.shape
            action = agent.step(state)[0]
            actions.append(action)
        actions = np.asarray(actions)
        batch_size = len(states)
        actions_shape = (batch_size,) + self._action_space.shape
        assert actions.shape == actions_shape, actions.shape
        return actions

    def episodes_end(self):
        for agent in self._agents:
            agent.episodes_end()


    def transitions(self, states, actions, rewards, next_states, term):
        assert len(states) == len(self._agents)
        stats = Statistics()
        for idx, agent in enumerate(self._agents):
            s = agent.transitions(
                np.expand_dims(states[idx], axis=0),
                np.expand_dims(actions[idx], axis=0),
                np.expand_dims(rewards[idx], axis=0),
                np.expand_dims(next_states[idx], axis=0),
                np.expand_dims(term[idx], axis=0),
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
