import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Arm():
    """
        Arm class that is used to sample from a beta prior
        Parameters:
            prior: a & b params for beta distribution
            trials: success and failure trials
    """
    def __init__(self):
        self.prior = np.ones(2) #start with uniform prior
        self.trials = np.zeros(2) #success, failure
        
    def moments(self):
        expectation = self.prior[0] / (self.prior[0] + self.prior[1])
        variance = (self.prior[0]*self.prior[1]) / (((self.prior[0] + self.prior[1])**2)*(np.sum(self.prior) - 1))
        return expectation, variance
    
    def update(self,response): #update prior
        arr = np.zeros(2)
        arr[1-response] = 1
        self.trials[1-response] += 1
        self.prior += arr
    
    def generate_sample_prior(self):
        return np.random.beta(self.prior[0],self.prior[1]) #sample