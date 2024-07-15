import numpy as np
import itertools

class BaseStation:
    def __init__(self, allocated_rbs, unit=5):
        self.allocated_rbs = allocated_rbs
        self.unit = unit
        self.actions = self.get_actions(allocated_rbs // unit, 3)
        self.action_size = len(self.actions)
        self.reset()

    def reset(self):
        self.allocated_resources = [0, 0, 0]  # Minimum allocation [eMBB, URLLC1, URLLC2]
        self.remaining_rbs = self.allocated_rbs - sum(self.allocated_resources)
        return self.get_state()

    def get_state(self):
        return self.allocated_resources + [self.remaining_rbs]

    def step(self, action_idx):
        num_users = 3
        action = self.actions[action_idx]
        if sum(action) > self.remaining_rbs or any(a < self.unit for a in action):
            self.allocated_resources = [rb + act for rb, act in zip(self.allocated_resources, [self.remaining_rbs // num_users + (1 if i < self.remaining_rbs % num_users else 0) for i in range(num_users)])]
            self.remaining_rbs = 0
        else:
            self.allocated_resources = [rb+act for rb, act in zip(self.allocated_resources, action)]
            self.remaining_rbs -= sum(action)
        reward = self.evaluate_qos()
        done = self.remaining_rbs == 0
        return self.get_state(), reward, done

    def evaluate_qos(self):
        # Calculate eMBB user rate using Shannon formula
        embb_rate = self.calculate_embb_rate(self.allocated_resources[0])
        urllc_delay1 = self.calculate_urllc_delay(self.allocated_resources[1])
        urllc_delay2 = self.calculate_urllc_delay(self.allocated_resources[2])

        # Check QoS requirements
        # if embb_rate >= 200 and urllc_delay1 <= 10 and urllc_delay2 <= 10:
        #     return 1  # Reward for meeting QoS
        # else:
        #     return -100  # Penalty for not meeting QoS
        return embb_rate-100+10-urllc_delay1+10-urllc_delay2

    def calculate_embb_rate(self, rbs):
        # Simplified Shannon formula, replace with actual formula if needed
        bandwidth = rbs
        snr = 5  # Example SNR value, replace with actual value if needed
        rate = bandwidth * np.log2(1 + snr)
        return rate

    def calculate_urllc_delay(self, rbs):
        # Simplified delay calculation, replace with actual calculation if needed
        rate = rbs * np.log2(1 + 5)
        delay = 100 / (rbs + 1)  # Example delay calculation
        return delay

    def get_actions(self, total_units, num_users):
        actions = []
        for action in itertools.product(range(1,total_units + 1), repeat=num_users):
            if sum(action) <= total_units:
                actions.append([a * self.unit for a in action])
        return actions
