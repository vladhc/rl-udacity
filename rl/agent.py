from rl import Reinforce, QLearning, ActorCritic, PPO, MultiPPO


def create_agent(env, args):

    action_space = env.action_space
    observation_shape = env.observation_space.shape
    print("Action space: {}".format(action_space))
    print("Observation space: {}".format(env.observation_space))

    agent_type = args["agent"]
    baseline = args["baseline"]
    baseline_learning_rate = args["baseline_learning_rate"]
    gamma = args["gamma"]
    learning_rate = args["learning_rate"]

    if agent_type == "qlearning":
        return QLearning(
                action_size=action_space.n,
                observation_shape=observation_shape,
                beta_decay=args["beta_decay"],
                gamma=gamma,
                learning_rate=learning_rate,
                soft=args["soft"],
                dueling=args["dueling"],
                double=args["double"],
                noisy=args["noisy"],
                priority=args["priority"],
                replay_buffer_size=args["replay_buffer_size"],
                min_replay_buffer_size=args["min_replay_buffer_size"],
                target_update_freq=args["target_update_freq"],
                train_freq=args["train_freq"],
                tau=args["tau"],
                batch_size=args["batch_size"],
                epsilon_start=args["epsilon_start"],
                epsilon_end=args["epsilon_end"],
                epsilon_decay=args["epsilon_decay"])
    elif agent_type == "reinforce":
        return Reinforce(
                action_size=action_space.n,
                observation_shape=observation_shape,
                gamma=gamma,
                learning_rate=learning_rate,
                baseline=baseline,
                baseline_learning_rate=baseline_learning_rate)
    elif agent_type == "actor-critic":
        return ActorCritic(
                action_size=action_space.n,
                observation_shape=observation_shape,
                gamma=gamma,
                learning_rate=learning_rate)
    elif agent_type == "ppo":
        return PPO(
                action_space=action_space,
                observation_shape=observation_shape,
                n_envs=env.n_envs,
                gamma=gamma,
                horizon=args["horizon"],
                epochs=args["ppo_epochs"],
                gae_lambda=args["gae_lambda"],
                learning_rate=learning_rate)
    elif agent_type == 'multippo':
        return MultiPPO(
                action_space=action_space,
                observation_shape=observation_shape,
                n_envs=env.n_envs,
                n_agents=env.n_agents,
                gamma=gamma,
                horizon=args["horizon"],
                epochs=args["ppo_epochs"],
                gae_lambda=args["gae_lambda"],
                learning_rate=learning_rate)
