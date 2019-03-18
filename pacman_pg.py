#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:28:32 2018

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

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Model
import keras.backend as K


#%% local functions

def smooth(x,n=10):
    window = np.ones(n)/n
    return np.convolve(x,window,'valid')
        
def get_lr(model):
    lr = model.optimizer.lr
    decay = model.optimizer.decay
    iterations = model.optimizer.iterations
    lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
    return K.eval(lr_with_decay)

def process_image(image):
    if image.ndim == 3:
        new_image = image/255.0
        new_image = new_image[np.newaxis,:]
    assert new_image.ndim == 4
    return new_image 

def listsum(grad_sum,grad,r = 1):
    n = len(grad)
    for i in range(n):
        grad_sum[i] = grad_sum[i] + r * grad[i]
    return grad_sum

#%% setup env


env = gym.make('MsPacman-v0')
n_actions = env.action_space.n
y = 0.9995
lr = 0.9




#%% define model

s_in = Input(shape = [210,160,3],dtype = 'float32')
conv1 = Conv2D(16,[3,3],padding = 'same',activation = 'relu')(s_in)
conv1 = MaxPooling2D(pool_size = [2,2],strides = 2)(conv1)
conv1 = Dropout(0.25)(conv1)
conv2 = Conv2D(32,[3,3],padding = 'same',activation = 'relu')(conv1)
conv2 = MaxPooling2D(pool_size = [2,2],strides = 2)(conv2)
conv2 = Dropout(0.25)(conv2)
conv3 = Conv2D(32,[3,3],padding = 'same',activation = 'relu')(conv2)
conv3 = MaxPooling2D(pool_size = [2,2],strides = 2)(conv3)
conv3 = Dropout(0.25)(conv3)
conv4 = Conv2D(64,[3,3],padding = 'same',activation = 'relu')(conv3)
conv4 = MaxPooling2D(pool_size = [2,2],strides = 2)(conv4)
conv4 = Dropout(0.25)(conv4)
f1 = Flatten()(conv4)
f1 = Dense(20,activation = 'tanh')(f1)
f1 = Dropout(0.5)(f1)
Q = Dense(n_actions, activation = 'softmax')(f1)


#%% simulate
model = Model(inputs = s_in,outputs = Q)
model.compile(optimizer = keras.optimizers.adam(lr = 0.01,decay = 1e-7),loss = 'mean_squared_error',metrics = [])

sess = tf.Session()
sess.run(tf.global_variables_initializer())


s = env.reset()
initial_weights = model.get_weights()
n_iterations = 10
n_samples = 16

# initialize grad_sum and r_hist
num_weights = len(initial_weights)
grad_sum = [np.zeros(w.shape) for w in initial_weights]
weighted_grad = [np.zeros(w.shape) for w in initial_weights]
r_hist = []
steps = 0


for iteration in range(n_iterations):
    print('running iteration {}...'.format(iteration))
    for sample in range(n_samples):
        print('  running sample {}...'.format(sample))
        for t in range(1000):
            if t%4 == 0:
                print('    running step {}...'.format(t))
                s = process_image(s)
                p_out = model.predict(s)
                a = np.random.choice(9, p = p_out[0]) # hack to make p_out 1d
    #            log_pi_theta = Q[:,a] # note batch dim
                grad = sess.run(K.gradients(Q, model.weights), feed_dict = {s_in: s})
                grad_sum = listsum(grad_sum,grad)
                s,r,d,log = env.step(a)
                r_hist.append(r)
                steps = steps+1
            else:
                s,r,d,log = env.step(a)
                r_hist.append(r)
        
        
        r_sum = sum(r_hist)
        weighted_grad = listsum(weighted_grad, grad_sum, r_sum)
        
        #reinitialize
        r_hist = []
        steps = 0
        grad_sum = [np.zeros(w.shape) for w in initial_weights]
        
        
    new_weights = listsum(model.get_weights(), weighted_grad)
    model.set_weights(new_weights)

    
    


