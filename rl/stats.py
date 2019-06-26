from collections import defaultdict
import numpy as np
import math
import tensorflow as tf


class Statistics(object):

    def __init__(self, summary_writer=None, iteration=None):
        self._dict = defaultdict(list)
        self._summary_writer = summary_writer
        self._iteration = iteration
        self._metrics = {
            'advantage': (self.avg, 'advantage'),
            'episode_reward': (self.avg, 'episode_reward'),
            'episode_steps': (self.avg, 'episode_steps'),
            'training_steps': (self.avg, 'training_steps'),
            'training_episodes': (self.avg, 'training_episodes'),
            'evaluation_episodes': (self.avg, 'eval_episodes'),
            'replay_buffer_beta': (self.avg, 'replay_beta'),
            'replay_buffer_size': (self.max, 'replay_buffer_size'),
            'replay_buffer_trajectories': (
                self.max, 'replay_buffer_trajectories'),
            'q': (self.avg, 'q'),
            'q_start': (self.avg, 'q_start'),
            'q_next_overestimate': (self.avg, 'q_next_overestimate'),
            'q_next_err': (self.avg, 'q_next_err'),
            'q_next_err_std': (self.avg, 'q_next_err_std'),
            'loss': (self.avg, 'loss'),
            'loss_actor': (self.avg, 'loss_actor'),
            'loss_critic': (self.avg, 'loss_critic'),
            'epsilon': (self.avg, 'epsilon'),
            'steps_per_second_env': (self.rate, 'env_time'),
            'steps_per_second': (self.rate, 'step_time'),
            'steps_per_second_optimization': (self.rate, 'optimization_time'),
            'ppo_optimization_epochs': (self.sum, 'ppo_optimization_epochs'),
            'ppo_optimization_samples': (self.avg, 'ppo_optimization_samples'),
            'noise_value_fc1': (self.avg, 'noise_value_fc1'),
            'noise_value_fc2': (self.avg, 'noise_value_fc2'),
            'noise_advantage_fc1': (self.avg, 'noise_advantage_fc1'),
            'noise_advantage_fc2': (self.avg, 'noise_advantage_fc2'),
            'noise_fc1': (self.avg, 'noise_fc1'),
            'noise_fc2': (self.avg, 'noise_fc2'),
            'return': (self.avg, 'return'),
            'baseline': (self.avg, 'baseline'),
            'entropy': (self.avg, 'entropy'),
            'grad_max': (self.max, 'grad_max'),
            'grad_mean': (self.avg, 'grad_mean'),
            'kl': (self.avg, 'kl'),
            'action_variance': (self.avg, 'action_variance'),
            'action_mu_mean': (self.avg, 'action_mu_mean'),
            'action_mu_max': (self.avg, 'action_mu_max'),
        }

    def set(self, key, value):
        k = self._dict[key]
        if isinstance(value, list):
            k.extend(value)
        else:
            k.append(value)
        self._dict[key] = k

    def get(self, arg):
        if isinstance(arg, dict):
            d = {}
            for old_key, new_key in arg.items():
                if old_key in self._dict:
                    d[new_key] = self._dict[old_key]
            return d
        if isinstance(arg, list):
            d = {}
            for key in arg:
                d[key] = self._dict[key]
            return d
        if isinstance(arg, str):
            return self._dict[arg]

    def set_all(self, props):
        if isinstance(props, Statistics):
            props = props._dict
        for key, val in props.items():
            self.set(key, val)

    def avg(self, key):
        if len(self._dict[key]) == 0:
            return 0.0
        return np.average(self._dict[key])

    def sum(self, key):
        return sum(self._dict[key])

    def max(self, key):
        if key not in self._dict:
            return None
        return max(self._dict[key])

    def count(self, key):
        return len(self._dict[key])

    def rate(self, key):
        s = self.sum(key)
        if s == 0:
            return None
        return self.count(key) / s

    def log(self):
        for metric, params in self._metrics.items():
            fn, key = params
            if key in self._dict:
                self._log_scalar(metric, fn(key))

    def _log_scalar(self, tag, value):
        if value is None or math.isnan(value):
            return
        s = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self._summary_writer.add_summary(s, self._iteration)
