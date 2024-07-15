import matplotlib.pyplot as plt
import numpy as np
def smooth_rewards(rewards, window_size):
    smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    return smoothed_rewards
data = np.load('rewards.npy')
plt.figure()
plt.plot(smooth_rewards(data, 20))
plt.savefig('SDN奖励.png')
plt.show()

data = np.load('losses.npy')
plt.figure()
plt.plot(smooth_rewards(data, 20))
plt.savefig('SDN损失.png')
plt.show()

data = np.load('rewards_BS.npy')
plt.figure()
plt.plot(smooth_rewards(data, 5))
plt.savefig('最后一次训练的基站的奖励.png')
plt.show()

data = np.load('losses_BS.npy')
plt.figure()
plt.plot(smooth_rewards(data, 2))
plt.savefig('最后一次训练的基站的损失.png')
plt.show()

data = np.load('bs_0_avg_rewards.npy')
plt.figure()
plt.plot(smooth_rewards(data, 20))
plt.savefig('bs_0训练中的平均奖励.png')
plt.show()