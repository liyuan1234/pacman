#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 12:46:38 2018

@author: liyuan
"""

import gym
import numpy as np
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
import time
import keras
import tensorflow as tf

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Model
import keras.backend as K


#%% local functions

def smooth(x,n):
    window = np.ones(n)/n
    return np.convolve(x,window,'same')
        
def get_lr(model):
    lr = model.optimizer.lr
    decay = model.optimizer.decay
    iterations = model.optimizer.iterations
    lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
    return K.eval(lr_with_decay)

#%% setup env


env = gym.make('MsPacman-v0')
n_actions = env.action_space.n


for i in range(100):
    env.reset()
    r = 0
    for j in range(10000):
        a = env.action_space.sample()
        for step in range(10):
            _,r_step,d,_ = env.step(a)  
            r = r+r_step
        env.render()
        time.sleep(0.05)
        if d == 1:
            print(r)
            break
            