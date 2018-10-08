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
    new_image = image/255.0
    new_image = new_image[np.newaxis,:]
    return new_image 

#%% setup env


env = gym.make('MsPacman-v0')
n_actions = env.action_space.n
y = 0.9995
lr = 0.9

jList = []
rList = []



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
Q_predict = Dense(n_actions)(f1)



#%% simulate
model = Model(inputs = s_in,outputs = Q_predict)
model.compile(optimizer = keras.optimizers.adam(lr = 0.01,decay = 1e-7),loss = 'mean_squared_error',metrics = [])

#print(model.summary())

render = 0
render_detail = 0

for i in range(10000):
    start_time = time.time()
    s = env.reset()
    a = env.action_space.sample()
    Q = model.predict(process_image(s))
    r_final = 0
    rand_action_flag = 0
    for j in range(10000):
#         if j%10000 == 0:
#             print('\trunning step {}...'.format(j))
        r = 0
        for frame in range(4):
            _,r_temp,_,_ = env.step(a)
            r = r + r_temp
        s1,r_temp,d,log = env.step(a)
        r = r+r_temp
        s1 = s1.astype(np.float32)
        Q1 = model.predict(process_image(s1))
        
        Q_target = copy.copy(Q)
        Q_target[:,a] = Q_target[:,a]+lr*(r+y*np.max(Q1) - Q_target[:,a])
        
        if render_detail == 1:
            print('Q and Q_target:')
            print(Q)
            print(Q_target)
            print(Q_target-Q)
            print(r)
            if rand_action_flag == 1:
                print('taking random action...')
                rand_action_flag = 0
            input()

        #s_gray = np.mean(s,axis = 2)
        
        model.fit(x = process_image(s),y = Q_target,epochs = 1,verbose = False)
        
        r_final += r
        s = s1
        Q = Q1
        a = np.argmax(Q)
        if np.random.random()>0.8:
            a = env.action_space.sample()
            rand_action_flag = 1
    
        if render == 1:
            env.render()

#            print(r)
            pass
        if d == 1:
            if r == 0:
    #            print('fell into hole!')
                pass
            if r == 1:
    #            print('success!')
    #            print(sess.run(W))
    #            input()
                pass
            
            jList.append(j)
            rList.append(r_final)
            current_lr = get_lr(model)
            
            print('running iteration {:>3}, number of steps is {:>5}, final score is: {:>8}, time taken to run session is {:.1f}, current learning rate is {:.5f}'.format(i,j*5,r_final,time.time() - start_time,current_lr))
            break



