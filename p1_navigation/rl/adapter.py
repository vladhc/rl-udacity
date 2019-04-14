class UnityEnvAdapter:

    def __init__(self, unity_env):
        self._env = unity_env
        self._brain_name = self._env.brain_names[0]

        # reset the environment
        env_info = self._env.reset(train_mode=False)[self._brain_name]

        # number of actions
        brain = self._env.brains[self._brain_name]
        self.action_size = brain.vector_action_space_size
        print('Number of actions:', self.action_size)
        # examine the state space
        state = env_info.vector_observations[0]
        self.state_size = len(state)
        print('State size:', self.state_size)

    def step(self, action):
        """ return next_state, action, reward, None """
        env_info = self._env.step(action)[self._brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return next_state, reward, done, None

    def reset(self):
        """ return state """
        env_info = self._env.reset(train_mode=False)[self._brain_name]
        state = env_info.vector_observations[0]
        return state
