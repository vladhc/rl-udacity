import numpy as np
import torch


class Trajectory:

    def __init__(
            self,
            capacity, observation_shape, action_type, action_shape, env_idx):
        self._cursor = 0
        self.env_idx = env_idx
        self._capacity = capacity
        # +1 here because we store states + one last state
        self._states = np.zeros(
                (capacity + 1,) + observation_shape, dtype=np.float16)
        self.actions = np.zeros(
                (capacity,) + action_shape, dtype=action_type)
        self.rewards = np.zeros(capacity, dtype=np.float16)
        self.terminated = False  # if the final state is term state

    def push(self, state, action, reward, next_state, done):
        assert not self.terminated
        assert self._cursor < self._capacity

        idx = self._cursor

        self._states[idx] = state
        self._states[idx+1] = next_state
        assert action.shape == self.actions[idx].shape
        self.actions[idx] = action
        self.rewards[idx] = reward

        self._cursor += 1

        if done:
            self.terminated = True
            self.close()

    def save(self):
        return {
            'states': self._states,
            'actions': self.actions,
            'rewards': self.rewards,
            'terminated': self.terminated,
            'env_idx': self.env_idx,
        }

    @staticmethod
    def load(d):
        t = Trajectory(
                1,
                observation_shape=d['observation_shape'],
                action_type=d['action_type'],
                action_shape=d['action_shape'],
                env_idx=d['env_idx'])
        t._states = d['states']
        t.actions = d['actions']
        t.rewards = d['rewards']
        t.terminated = d['terminated']
        t._cursor = len(t.actions)
        return t

    @property
    def states(self):
        return self._states[:self._cursor]

    @property
    def next_states(self):
        return self._states[1:self._cursor+1]

    def close(self):
        # +1 here because we store states + one last state
        self._states = self._states[:self._cursor + 1, :]
        self.actions = self.actions[:self._cursor]
        self.rewards = self.rewards[:self._cursor]

    def done(self):
        return self.terminated or self._capacity == len(self)

    def __len__(self):
        return self._cursor


class TrajectoryBuffer:

    def __init__(
            self,
            observation_shape,
            action_space):
        self._trajectories = dict()
        self._observation_shape = observation_shape
        self._action_space = action_space
        self.reset()

    def _create_trajectory(self, env_idx):
        return Trajectory(
            10000,
            observation_shape=self._observation_shape,
            action_type=self._action_space.dtype,
            action_shape=self._action_space.shape,
            env_idx=env_idx)

    def reset(self):
        self._records_collected = 0
        self.trajectories = []
        self._trajectories.clear()

    def save(self, filename):
        self._finish_trajectories()
        torch.save(
            {
                "memory_store": [
                    traj.save() for traj in self.trajectories],
                "observation_shape": self._observation_shape,
                "action_space": self._action_space,
            },
            filename)

    @staticmethod
    def load(filename):
        d = torch.load(filename)
        b = TrajectoryBuffer(
                observation_shape=d["observation_shape"],
                action_space=d["action_space"])
        memory = d["memory_store"]
        while len(memory) != 0:
            traj_dict = memory.pop(0)
            traj_dict['observation_shape'] = b._observation_shape
            traj_dict['action_type'] = b._action_space.dtype
            traj_dict['action_shape'] = b._action_space.shape
            traj = Trajectory.load(traj_dict)
            b._append(traj)
        return b

    def push(self, states, actions, rewards, next_states, dones):
        for idx in range(len(states)):
            self._push_single(
                    states[idx],
                    actions[idx],
                    rewards[idx],
                    next_states[idx],
                    dones[idx],
                    idx)

    def _push_single(self, state, action, reward, next_state, done, env_idx):
        if env_idx not in self._trajectories:
            self._trajectories[env_idx] = self._create_trajectory(env_idx)
        traj = self._trajectories[env_idx]
        traj.push(state, action, reward, next_state, done)
        self._trajectories[env_idx] = traj
        if traj.done():
            self._append(traj)
            del self._trajectories[env_idx]

    def _append(self, traj):
        self.trajectories.append(traj)
        self._records_collected += len(traj)

    def __len__(self):
        return self._records_collected + \
                sum([len(traj) for traj in self._trajectories.values()])

    def _finish_trajectories(self):
        for _, traj in self._trajectories.items():
            if len(traj) > 0:
                traj.close()
                self._append(traj)
