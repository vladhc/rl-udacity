import torch
from rl import UnityEnvAdapter, GreedyPolicy
from unityagents import UnityEnvironment


def main():
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
    env = UnityEnvAdapter(env)
    q_net = torch.load('checkpoint-220.pth')
    q_net.train(False)

    while True:
        play_episode(env, q_net)

def play_episode(env, q_net):
    policy = GreedyPolicy()
    state = env.reset()
    reward_acc = 0
    while True:
        state_tensor = torch.from_numpy(state).unsqueeze(0)
        q_values = q_net(state_tensor)
        action = policy.get_action(q_values)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        reward_acc += reward
        if done:
            break
    print("Reward: {}".format(reward_acc))


if __name__ == '__main__':
    main()
