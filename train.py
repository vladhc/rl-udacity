import argparse
from google.cloud import storage

from rl import Runner, TrajectoryBuffer, create_env, create_agent


BUCKET = 'rl-1'


def main(**args):
    envs_count = args["env_count"]
    env = create_env(args["env"], envs_count)
    del args["env"]

    agent = create_agent(env, args)

    sess = args["sess"]
    sess += "-" + args["agent"]
    sess_options = [
            "double", "priority", "dueling",
            "noisy", "soft", "baseline"]
    for opt in sess_options:
        if args[opt]:
            sess += "-" + opt

    bucket = None
    gcp = args["gcp"]
    if gcp:
        client = storage.Client()
        bucket = client.get_bucket(BUCKET)

    evaluation_steps = args['eval_steps']
    training_steps = args['steps']
    iterations = args['iterations']

    traj_buffer = None
    if args["save_traj"]:
        traj_buffer = TrajectoryBuffer(
                observation_shape=observation_shape,
                action_space=action_space)
        print("Saving trajectories is enabled")

    runner = Runner(
            env,
            agent,
            sess,
            bucket=bucket,
            traj_buffer=traj_buffer,
            num_iterations=iterations,
            training_steps=training_steps,
            evaluation_steps=evaluation_steps)
    runner.run_experiment()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sess", type=str)
    parser.add_argument("--env", type=str)
    parser.add_argument("--env_count", type=int, default=1)
    parser.add_argument("--agent", type=str, default="qlearning",
            help="qlearning|reinforce|actor-critic")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--noisy", action="store_true",
            help="Enables noisy network")
    parser.add_argument("--priority", action="store_true",
            help="Enables prioritirized replay buffer")
    parser.add_argument("--soft", action="store_true",
            help="Enables soft update of target network")
    parser.add_argument("--baseline", action="store_true",
            help="Enables baseline for the REINFORCE agent.")
    parser.add_argument("--epsilon_decay", type=int, default=3000)
    parser.add_argument("--beta_decay", type=int, default=3000,
            help="How many steps should beta parameter decay. " +
            "Used in the Q-Learning")
    parser.add_argument("--steps", type=int, default=100,
            help="Number of steps for training phase in one iteration")
    parser.add_argument("--eval_steps", type=int, default=100,
            help="Number of steps for evaluation phase in one iteration")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--target_update_freq", type=int, default=100,
            help="Update target network each N steps")
    parser.add_argument("--epsilon_start", type=float, default=0.5)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--baseline_learning_rate", type=float, default=0.0001)
    parser.add_argument("--replay_buffer_size", type=int, default=100000,
            help="Maximum size of the replay buffer")
    parser.add_argument("--min_replay_buffer_size", type=int, default=128,
            help="Size of the replay buffer before optimization starts")
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--gcp", action="store_true",
            help="Sets if Google Cloud Platform storage bucket should " +
            "be used for storing training results.")
    parser.add_argument("--tau", type=float, default=0.001,
            help="Soft update parameter")
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=128,
        help="PPO parameter. How many timesteps collect experience " +
        "before starting optimization phase.")
    parser.add_argument("--ppo_epochs", type=int, default=12,
        help="PPO parameter. Epochs count in the optimization phase.")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
        help="lambda parameter for Advantage Function Estimation (GAE)")
    parser.add_argument("--save_traj", action="store_true",
            help="Enables persisting trajectories on disk during " +
            "training/execution time.")

    parser.set_defaults(dueling=False)
    parser.set_defaults(double=False)
    parser.set_defaults(noisy=False)
    parser.set_defaults(priority=False)
    parser.set_defaults(soft=False)
    parser.set_defaults(baseline=False)
    parser.set_defaults(gcp=False)
    parser.set_defaults(save_traj=False)
    args = parser.parse_args()

    d = vars(args)
    main(**d)
