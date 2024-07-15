import numpy as np
import itertools

class SDNController:
    def __init__(self, total_rbs, num_base_stations, unit=15):
        self.total_rbs = total_rbs
        self.num_base_stations = num_base_stations
        self.unit = unit
        self.actions = self.get_actions(total_rbs // unit, num_base_stations)
        self.action_size = len(self.actions)
        self.reset()

    def reset(self):
        self.allocated_rbs = [0] * self.num_base_stations
        self.remaining_rbs = self.total_rbs
        return self.get_state()

    def get_state(self):
        return self.allocated_rbs + [self.remaining_rbs]

    def step(self, action_idx, feedback=[]):
        action = self.actions[action_idx]
        allocated_rbs = sum(action)
        if allocated_rbs > self.remaining_rbs or any(a < self.unit for a in action):
            self.allocated_rbs = [rb + act for rb, act in zip(self.allocated_rbs, [5]*self.num_base_stations)]
            self.remaining_rbs -= 5*self.num_base_stations
        else:
            self.allocated_rbs = [rb + act for rb, act in zip(self.allocated_rbs, action)]
            self.remaining_rbs -= allocated_rbs
        reward = self.evaluate_qos(feedback)
        done = self.remaining_rbs <= 0
        return self.get_state(), reward, done

    def evaluate_qos(self, feedback):
        return sum(feedback)  if feedback else sum(self.allocated_rbs) / self.total_rbs

    def get_actions(self, total_units, num_stations):
        actions = []
        for action in itertools.product(range(1, total_units + 1), repeat=num_stations):
            if sum(action) <= total_units:
                actions.append([a * self.unit for a in action])
        return actions
