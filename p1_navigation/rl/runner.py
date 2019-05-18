import shutil
import sys
import time
import tensorflow as tf
from rl import Statistics
import math

from google.cloud import storage

BUCKET = 'rl-1'


class Runner(object):

    def __init__(
            self,
            env,
            agent,
            session_id,
            num_iterations,
            training_steps,
            evaluation_steps,
            max_episode_steps,
            gcp):

        self._env = env
        self._agent = agent
        self._session_id = session_id
        self._num_iterations = num_iterations
        self._iteration = 0
        self._training_steps = training_steps
        self._evaluation_steps = evaluation_steps
        self._max_episode_steps = max_episode_steps

        print("Session ID: {}".format(self._session_id))
        print("Iterations: {}".format(self._num_iterations))
        print("Training steps per iteration: {}".format(
            self._training_steps))
        print("Evaluation steps per iteration: {}".format(
            self._evaluation_steps))
        print("Maximum steps per episode: {}".format(
            self._max_episode_steps))

        self._bucket = None
        if gcp:
            client = storage.Client()
            self._bucket = client.get_bucket(BUCKET)

        out_dir = 'gs://{}'.format(BUCKET) if gcp else '.'
        summary_file = '{}/train/{}'.format(out_dir, self._session_id)
        shutil.rmtree(summary_file, ignore_errors=True)
        self._summary_writer = tf.summary.FileWriter(summary_file, None)

    def run_experiment(self):
        for iteration in range(self._num_iterations):
            self._iteration = (iteration + 1)
            statistics = self._run_one_iteration()
            self._log_experiment(statistics)
            self._checkpoint()

    def _log_experiment(self, stats):
        self._log_scalar('episode_reward', stats.avg('episode_reward'))
        self._log_scalar('episode_steps', stats.avg('episode_steps'))
        self._log_scalar('training_steps', stats.avg('training_steps'))
        self._log_scalar('training_episodes', stats.avg('training_episodes'))
        self._log_scalar('evaluation_episodes', stats.avg('eval_episodes'))
        self._log_scalar('replay_buffer_beta', stats.avg('replay_beta'))
        self._log_scalar('replay_buffer_size', stats.max('replay_buffer_size'))
        self._log_scalar('q', stats.avg('q'))
        self._log_scalar('q_start', stats.avg('q_start'))
        self._log_scalar('loss', stats.avg('loss'))
        self._log_scalar('epsilon', stats.avg('epsilon'))
        self._log_scalar('steps_per_second_env', stats.rate('env_time'))
        self._log_scalar('steps_per_second', stats.rate('step_time'))
        self._log_scalar(
                'steps_per_second_optimization',
                stats.rate('optimization_time'))
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
        filename = '{}-{}.pth'.format(self._session_id, self._iteration)
        path = './checkpoints/{}'.format(filename)
        self._agent.save_model(path)
        if self._bucket:
            blob = self._bucket.blob(
                    'checkpoints/{}'.format(
                        filename,
                        self._session_id,
                        self._iteration))
            blob.upload_from_filename(filename=path)

    def _run_one_iteration(self):
        stats = Statistics()

        rewards, steps = self._run_one_phase(stats, is_training=True)
        stats.set('training_episodes', len(steps))
        stats.set('training_steps', sum(steps))

        if self._evaluation_steps != 0:
            rewards, steps = self._run_one_phase(stats, is_training=False)
            stats.set('eval_episodes', len(steps))
        stats.set_all('episode_reward', rewards)
        stats.set_all('episode_steps', steps)

        return stats

    def _run_one_phase(self, stats, is_training):
        rewards = []
        steps = []
        self._agent.eval = not is_training
        min_steps = self._training_steps if is_training \
            else self._evaluation_steps
        while sum(steps) < min_steps:
            step, reward = self._run_one_episode(stats)
            steps.append(step)
            rewards.append(reward)

            sys.stdout.write('Iteration {} ({}). '.format(
                                        self._iteration,
                                        "train" if is_training else "eval") +
                             'Steps executed: {} '.format(sum(steps)) +
                             'Episode length: {} '.format(steps[-1]) +
                             'Return: {:.2f}      \r'.format(rewards[-1]))
            sys.stdout.flush()
        print()
        return rewards, steps

    def _run_one_episode(self, stats):

        # For logging purposes
        episode_steps = 0
        reward_acc = 0.0

        state = self._env.reset()
        reward = None

        is_training = not self._agent.eval

        while True:
            step_time0 = time.time()

            t0 = time.time()
            action = self._agent.step(state, reward, stats)
            if is_training:
                stats.set('agent_time', time.time() - t0)

            t0 = time.time()
            next_state, reward, done, _ = self._env.step(action)
            if is_training:
                stats.set('env_time', time.time() - t0)

            if done:
                next_state = None
            state = next_state

            reward_acc += reward
            episode_steps += 1

            if is_training:
                stats.set('step_time', time.time() - step_time0)

            if done or episode_steps == self._max_episode_steps:
                break

        self._agent.end_episode(reward)
        return episode_steps, reward_acc
