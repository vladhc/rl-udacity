import argparse

from rl import Runner, create_env
from rl import Reinforce, QLearning

from google.cloud import storage

BUCKET = 'rl-1'


def main(**args):
    env = create_env(args['env'])
    del args['env']

    iterations = args['iterations']
    del args['iterations']
    training_steps = args['steps']
    del args['steps']
    max_episode_steps = args['max_episode_steps']
    del args['max_episode_steps']
    evaluation_steps = args['eval_steps']
    del args['eval_steps']
    gcp = args['gcp']
    del args['gcp']

    sess = args['sess']
    sess += '-' + args['agent']
    sess_options = ['double', 'priority', 'dueling', 'noisy', 'soft']
    for opt in sess_options:
        if args[opt]:
            sess += '-' + opt
    del args['sess']

    try:
        action_size = env.action_space.n
        observation_shape = env.observation_space.shape
    except AttributeError:
        action_size = env.action_size
        observation_shape = (env.state_size, )

    ref_net = args['ref_net']

    bucket = None
    if gcp:
        client = storage.Client()
        bucket = client.get_bucket(BUCKET)
        if ref_net is not None:
            if not ref_net.endswith(".pth"):
                ref_net += ".pth"
            ref_net = "checkpoints/{}".format(ref_net)
            args['ref_net'] = ref_net
            blob = storage.Blob(ref_net, bucket)
            with open(ref_net, "wb") as f:
                blob.download_to_file(f)

    agent_type = args["agent"]
    del args["agent"]

    if agent_type == "qlearning":
        agent = QLearning(
                action_size=action_size,
                observation_shape=observation_shape,
                beta_decay=(iterations * training_steps),
                **args)
    elif agent_type == "reinforce":
        gamma = args['gamma']
        learning_rate = args['learning_rate']
        agent = Reinforce(
                action_size=action_size,
                observation_shape=observation_shape,
                gamma=gamma,
                learning_rate=learning_rate)

    runner = Runner(
            env,
            agent,
            sess,
            bucket=bucket,
            num_iterations=iterations,
            training_steps=training_steps,
            evaluation_steps=evaluation_steps,
            max_episode_steps=max_episode_steps)
    runner.run_experiment()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sess")
    parser.add_argument("--env")
    parser.add_argument("--agent", type=str, default="qlearning",
            help="qlearning|reinforce")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--noisy", action="store_true",
            help="Enables noisy network")
    parser.add_argument("--priority", action="store_true",
            help="Enables prioritirized replay buffer")
    parser.add_argument("--soft", action="store_true",
            help="Enables soft update of target network")
    parser.add_argument("--epsilon_decay", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=100,
            help="Number of steps for training phase in one iteration")
    parser.add_argument("--eval_steps", type=int, default=100,
            help="Number of steps for evaluation phase in one iteration")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--max_episode_steps", type=int, default=2000)
    parser.add_argument("--target_update_freq", type=int, default=100,
            help="Update target network each N steps")
    parser.add_argument("--epsilon_start", type=float, default=0.5)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
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
    parser.add_argument("--ref_net", type=str,
        help="Used for debugging of Q values overestimation. " +
        "This checkpoint should point to an already trained network. " +
        "The network is used for extimation of V_next* " +
        "(true next state values).")

    parser.set_defaults(dueling=False)
    parser.set_defaults(double=False)
    parser.set_defaults(noisy=False)
    parser.set_defaults(priority=False)
    parser.set_defaults(soft=False)
    parser.set_defaults(gcp=False)
    args = parser.parse_args()

    d = vars(args)
    main(**d)
