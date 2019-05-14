import shutil
import sys
import time
import tensorflow as tf
from rl import Statistics
import math


class Runner(object):

    def __init__(
            self,
            env,
            agent,
            session_id,
            num_iterations,
            training_steps,
            max_episode_steps,
            training_done_fn=lambda x: False):

        self._env = env
        self._agent = agent
        self._session_id = session_id
        self._num_iterations = num_iterations
        self._iteration = 0
        self._training_steps = training_steps
        self._max_episode_steps = max_episode_steps
        self._training_done_fn = training_done_fn  # Not used so far

        summary_file = "./train/{}".format(self._session_id)
        shutil.rmtree(summary_file, ignore_errors=True)
        self._summary_writer = tf.summary.FileWriter(summary_file, None)

    def run_experiment(self):
        print('Starting training...')
        for iteration in range(self._num_iterations):
            self._iteration = (iteration + 1)
            statistics = self._run_one_iteration()
            self._log_experiment(statistics)
            self._checkpoint()

    def _log_experiment(self, stats):
        self._log_scalar('episode_reward', stats.avg('episode_reward'))
        self._log_scalar('episode_steps', stats.avg('episode_steps'))
        self._log_scalar('steps', stats.avg('iteration_steps'))
        self._log_scalar('episodes', stats.avg('iteration_episodes'))
        self._log_scalar('replay_buffer_beta', stats.avg('replay_beta'))
        self._log_scalar('replay_buffer_size', stats.max('replay_buffer_size'))
        self._log_scalar('q', stats.avg('q'))
        self._log_scalar('q_start', stats.avg('q_start'))
        self._log_scalar('loss', stats.avg('loss'))
        self._log_scalar('epsilon', stats.avg('epsilon'))
        self._log_scalar('step_time', stats.avg('step_time'))
        self._log_scalar('env_time', stats.avg('env_time'))
        self._log_scalar('agent_time', stats.avg('agent_time'))
        self._log_scalar(
                'optimization_time',
                stats.avg('optimization_time'))
        self._log_scalar('noise_value_fc1', stats.avg('noise_value_fc1'))
        self._log_scalar('noise_value_fc2', stats.avg('noise_value_fc2'))
        self._log_scalar(
                'noise_advantage_fc1',
                stats.avg('noise_advantage_fc1'))
        self._log_scalar(
                'noise_advantage_fc2',
                stats.avg('noise_advantage_fc2'))
        self._log_scalar('noise_fc1', stats.avg('noise_fc1'))
        self._log_scalar('noise_fc2', stats.avg('noise_fc2'))
        self._summary_writer.flush()

    def _log_scalar(self, tag, value):
        if math.isnan(value):
            return
        s = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self._summary_writer.add_summary(s, self._iteration)

    def _checkpoint(self):
        filename = 'checkpoints/{}-{}.pth'.format(
                self._session_id,
                self._iteration)
        self._agent.save_model(filename)

    def _run_one_iteration(self):
        print("Starting iteration {}".format(self._iteration))
        stats = Statistics()
        step_count = 0
        num_episodes = 0
        sum_reward = 0

        while step_count < self._training_steps:
            steps, reward = self._run_one_episode(stats)
            stats.set('episode_reward', reward)
            stats.set('episode_steps', reward)
            step_count += steps
            sum_reward += reward
            num_episodes += 1

            sys.stdout.write('Steps executed: {} '.format(step_count) +
                             'Episode length: {} '.format(steps) +
                             'Return: {}\n'.format(reward))
            sys.stdout.flush()

        stats.set('iteration_episodes', num_episodes)
        stats.set('iteration_steps', step_count)

        return stats

    def _run_one_episode(self, stats):

        # For logging purposes
        episode_steps = 0
        reward_acc = 0.0
        env_time = 0.0  # time spent for experience generation

        state = self._env.reset()
        reward = None

        while True:
            step_time0 = time.time()

            t0 = time.time()
            action = self._agent.step(state, reward, stats)
            stats.set('agent_time', time.time() - t0)

            t0 = time.time()
            next_state, reward, done, _ = self._env.step(action)
            stats.set('env_time', time.time() - t0)

            if done:
                next_state = None
            state = next_state

            reward_acc += reward
            episode_steps += 1

            stats.set('step_time', time.time() - t0)

            if done or episode_steps == self._max_episode_steps:
                break

        self._agent.end_episode(reward)
        return episode_steps, reward_acc
