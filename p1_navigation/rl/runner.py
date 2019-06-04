import shutil
import sys
import time
import numpy as np
import tensorflow as tf
from rl import Statistics


class Runner(object):

    def __init__(
            self,
            env,
            agent,
            session_id,
            num_iterations,
            training_steps,
            evaluation_steps,
            bucket):

        self._env = env
        self._agent = agent
        self._session_id = session_id
        self._num_iterations = num_iterations
        self._iteration = 0
        self._training_steps = training_steps
        self._evaluation_steps = evaluation_steps

        print("Session ID: {}".format(self._session_id))
        print("Iterations: {}".format(self._num_iterations))
        print("Training steps per iteration: {}".format(
            self._training_steps))
        print("Evaluation steps per iteration: {}".format(
            self._evaluation_steps))

        self._bucket = bucket

        out_dir = 'gs://{}'.format(bucket.name) if bucket is not None else '.'
        summary_file = '{}/train/{}'.format(out_dir, self._session_id)
        shutil.rmtree(summary_file, ignore_errors=True)
        self._summary_writer = tf.summary.FileWriter(summary_file, None)

    def run_experiment(self):
        for iteration in range(self._num_iterations):
            self._iteration = (iteration + 1)
            statistics = self._run_one_iteration()
            statistics.log()
            self._checkpoint()

    def _checkpoint(self):
        filename = '{}-{}.pth'.format(self._session_id, self._iteration)
        path = './checkpoints/{}'.format(filename)
        self._agent.save_model(path)
        if self._bucket:
            blob = self._bucket.blob(
                    'checkpoints/{}'.format(filename))
            blob.upload_from_filename(filename=path)

    def _run_one_iteration(self):
        stats = Statistics(self._summary_writer, self._iteration)

        phase_stats, agent_stats = self._run_one_phase(is_training=True)
        stats.set("training_episodes", phase_stats.sum("episodes"))
        stats.set("training_steps", phase_stats.sum("steps"))
        stats.set_all(phase_stats.get(["agent_time", "step_time", "env_time"]))
        stats.set_all(agent_stats)

        if self._evaluation_steps != 0:
            phase_stats, _ = self._run_one_phase(is_training=False)
            stats.set("eval_episodes", phase_stats.sum("episodes"))
        stats.set("episode_reward", phase_stats.get("rewards"))
        stats.set("episode_steps", phase_stats.get("steps"))

        return stats

    def _run_one_phase(self, is_training):
        stats = Statistics()
        agent_stats = Statistics()

        self._agent.eval = not is_training
        min_steps = self._training_steps * self._env.n_agents if is_training \
            else self._evaluation_steps

        self._env.reset()
        while stats.sum("steps") < min_steps:
            step_time0 = time.time()

            states = np.copy(self._env.states)
            actions = self._agent.step(states)

            rewards, next_states, dones, env_stats = \
                self._env.step(actions)
            stats.set_all(env_stats)

            if is_training:
                t0 = time.time()
                agent_stats.set_all(
                        self._agent.transitions(
                            states,
                            actions,
                            rewards,
                            next_states,
                            dones))
                stats.set("agent_time", time.time() - t0)
                stats.set("step_time", time.time() - step_time0)

            sys.stdout.write("Iteration {} ({}). ".format(
                                        self._iteration,
                                        "train" if is_training else "eval") +
                             "Steps executed: {} ".format(stats.sum("steps")) +
                             "Episode length: {} ".format(
                                 int(stats.avg("steps"))) +
                             "Return: {:.2f}      \r".format(
                                 stats.avg("rewards")))
            sys.stdout.flush()
        print()
        self._agent.episodes_end()
        return stats, agent_stats
