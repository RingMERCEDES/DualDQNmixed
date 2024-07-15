import torch
import torch.optim as optim
from dqn import DQN, ReplayMemory, optimize_model
from sdn_controller import SDNController
from base_station import BaseStation
import random
import matplotlib.pyplot as plt
import numpy as np
def smooth_rewards(rewards, window_size):
    smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    return smoothed_rewards

def train_agent():
    sdn_env = SDNController(TOTAL_RBS, NUM_BASE_STATIONS, UNIT_SDN)
    policy_net = DQN(STATE_SIZE_SDN, sdn_env.action_size)
    target_net = DQN(STATE_SIZE_SDN, sdn_env.action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(MEMORY_SIZE)
    epsilon = EPSILON_START

    rewards = []
    losses = []
    all_feedback = []
    all_allocations = []
    all_bs_rewards = [[] for _ in range(NUM_BASE_STATIONS)]
    all_bs_avg_rewards = [[] for _ in range(NUM_BASE_STATIONS)]
    feedback = []
    episodes = 3000
    for episode in range(episodes):
        state = sdn_env.reset()
        done = False
        total_reward = 0
        episode_loss = 0
        # all_bs_rewards = [[] for _ in range(NUM_BASE_STATIONS)]
        # feedback = []
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if random.random() < epsilon:
                action_idx = random.randint(0, sdn_env.action_size - 1)
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action_idx = q_values.argmax(1).item()

            next_state, reward, done = sdn_env.step(action_idx, feedback)
            memory.push(state, action_idx, reward, next_state, done)
            loss = optimize_model(policy_net, target_net, memory, optimizer, BATCH_SIZE, GAMMA)

            state = next_state
            total_reward += reward
            episode_loss += loss if loss is not None else 0

        # Train base stations and collect feedback
        feedback.clear()
        for i in range(NUM_BASE_STATIONS):
            alloRBS = state[i]
            avg_reward, bs_rewards = train_bs_agent(alloRBS)
            feedback.append(avg_reward)
            all_bs_avg_rewards[i].append(avg_reward)
            all_bs_rewards[i].extend(bs_rewards)

        all_feedback.append(feedback)
        all_allocations.append(state[:-1])  # Exclude the remaining RBs part from state

        epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon)
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards.append(total_reward)
        losses.append(episode_loss)

        print(f'Episode {episode}, Total Reward: {total_reward}, Loss: {episode_loss}')

    # Save data
    np.save("rewards.npy", rewards)
    np.save("losses.npy", losses)
    np.save("feedback.npy", all_feedback)
    np.save("allocations.npy", all_allocations)
    for i in range(NUM_BASE_STATIONS):
        # np.save(f"bs_{i}_rewards.npy", all_bs_rewards[i])   # all_bs_rewards记录了BS训练中的所有奖励的变化，大小为两个eipsodes相乘
        np.save(f"bs_{i}_avg_rewards.npy", all_bs_avg_rewards[i])  #这个记录了3000个episodes中，BS的reward变化

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(rewards)
    plt.title('SDN Controller Training Results')
    plt.ylabel('Total Reward')
    plt.subplot(2, 1, 2)
    plt.plot(smooth_rewards(rewards, 50))
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Total Reward')
    plt.savefig('sdn_training_rewards.png')
    plt.show()

    # for i in range(NUM_BASE_STATIONS):
    #     plt.figure(figsize=(12, 8))
    #     plt.plot(all_bs_rewards[i])
    #     plt.title(f'Base Station {i} Training Rewards')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Total Reward')
    #     plt.savefig(f'bs_{i}_training_rewards.png')
    #     plt.show()


def train_bs_agent(allocated_rbs):
    bs_env = BaseStation(allocated_rbs, UNIT_BS)
    policy_net = DQN(STATE_SIZE_BS, bs_env.action_size)
    target_net = DQN(STATE_SIZE_BS, bs_env.action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(MEMORY_SIZE)
    epsilon = EPSILON_START

    episodes = 1000
    total_rewards = []
    losses = []

    for episode in range(episodes):
        state = bs_env.reset()
        done = False
        total_reward = 0
        episode_loss = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if random.random() < epsilon:
                action_idx = random.randint(0, bs_env.action_size - 1)
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action_idx = q_values.argmax(1).item()

            next_state, reward, done = bs_env.step(action_idx)
            memory.push(state, action_idx, reward, next_state, done)
            loss = optimize_model(policy_net, target_net, memory, optimizer, BATCH_SIZE, GAMMA)

            state = next_state
            total_reward += reward
            episode_loss += loss if loss is not None else 0

        epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon)
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        total_rewards.append(total_reward)
        losses.append(episode_loss)
    np.save("rewards_BS.npy", total_rewards)  # 这里的total_rewards是一个记录了BS episodes的奖励的列表
    np.save("losses_BS.npy", losses)         # 这里的losses是一个记录了BS episodes的loss的列表
    avg_reward = np.mean(total_rewards[-100:])
    return avg_reward, total_rewards           # 返回的是训练完所有episodes后的一个最终达到的平均奖励  和 训练中的所有记录的奖励


if __name__ == '__main__':
    # Hyperparameters
    TOTAL_RBS = 450
    NUM_BASE_STATIONS = 3
    UNIT_SDN = 15
    UNIT_BS = 5
    STATE_SIZE_SDN = NUM_BASE_STATIONS + 1
    STATE_SIZE_BS = 4
    MEMORY_SIZE = 10000
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE = 20

    train_agent()

