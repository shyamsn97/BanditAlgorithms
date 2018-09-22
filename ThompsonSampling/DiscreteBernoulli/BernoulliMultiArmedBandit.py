import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import beta
from Arm import Arm

class BernoulliMultiArmedBandit():
    """
        Main Bernoulli Multi Armed Bandit class
        Uses Thompson sampling to greedily choose arms by picking the one with the highest sampled prior.
        Epsilon Greedy strategy is also added for exploration
        Parameters:
            data: two columns, 1st is bandit ids, 2nd is binary denoting success
            epsilon: exploration value
            decay: exploration decay
            num_arms: number of arms
            indices: indices for each arm
            sampled_priors: sampled priors for each arm
            arms: Arm() objects
            final_values: final values
    """
    def __init__(self,data,num_arms,epsilon=0.9,decay=0.95):
        self.epsilon = epsilon
        self.decay = decay
        self.data = data
        self.num_arms = num_arms
        self.indices = self.generate_indices()
        self.sampled_priors = np.zeros(self.num_arms)
        self.arms = [Arm() for i in range(num_arms)]
        self.final_values = {}
    
    def reset(self):
        self.sampled_priors = np.zeros(self.num_arms)
        self.arms = [Arm() for i in range(self.num_arms)]
        self.final_values = {}
    
    def generate_indices(self):
        indices = {}
        for i in range(self.num_arms):
            indices[i] = np.where(self.data[:,0] == i)[0]
        return indices
    
    def return_final_values(self):
        for i in range(self.num_arms):
            arm = self.arms[i]
            prior = arm.prior
            expectation, variance = arm.moments()
            vals = [prior[0],prior[1],expectation,variance]
            self.final_values[i] = vals
        
    def act(self):
        if np.random.rand() < self.epsilon:
            self.epsilon *= self.decay
            arm_index = np.random.choice(self.num_arms)
        else:
            for i in range(self.num_arms):
                arm = self.arms[i]
                self.sampled_priors[i] = arm.generate_sample_prior()
            arm_index = np.argmax(self.sampled_priors)
        arm = self.arms[arm_index]
        response = self.data[np.random.choice(self.indices[arm_index])][-1]
        arm.update(response)
        
    def train(self,iterations=1000):
        self.reset()
        bar = tqdm(np.arange(iterations))
        for i in bar:
            self.act()
            bar.set_description("Epsilon: %s" % str(self.epsilon))
        self.return_final_values()

