# -*- coding: utf-8 -*-
"""
Created on Wed July 13 2022

@author: Junchao
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
from collections import deque
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.python.framework import ops
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

tf.compat.v1.disable_eager_execution()

        
class DQNAgent:
    def __init__(self, state_size, n_rows, n_cols, q_state_size, action_size, gamma, gammaB):
        self.state_size = state_size
        self.q_state_size = q_state_size
        self.action_size = action_size
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.memory = deque(maxlen=int(1e6)) # =1000000
        self.gamma = gamma    # discount rate
        self.gammaB = gammaB    # discount rate B
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.998 #0.995
        self.learning_rate = 0.001
        self.model_tar = self._build_model() # target network
        self.model_eval = self._build_model() # evaluation network
    
    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(8, (4, 4), strides=(2, 2), activation='relu', input_shape=(self.n_rows, self.n_cols, self.q_state_size)))
        model.add(tf.keras.layers.Conv2D(16, (2, 2), strides=(1, 1), activation='relu'))
        model.add(tf.keras.layers.Flatten())
        
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            arr = np.random.rand(self.action_size)
            return arr
        act_q_value = self.model_tar.predict(state)
        return act_q_value  # returns action
    
    def act_trained(self, state):
        # Select the action for using the model
        act_q_value = self.model_tar.predict(state)
        return act_q_value  # returns action
    
    def replay(self, batch_size, num_episode, episode):
        minibatch = random.sample(self.memory, batch_size)
        train_X = []
        train_Y = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model_tar.predict(next_state)))
            else:
                target = (reward + self.gammaB * np.amax(self.model_tar.predict(next_state)))
            target_f = self.model_eval.predict(state)
            target_f[0][np.argmax(action)] = target
            train_X.append(state)
            train_Y.append(target_f[0])
            
            arr_train_X = np.array(train_X)
            arr_train_Y = np.array(train_Y)
            arr_train_X = np.reshape(arr_train_X, (-1,self.n_rows, self.n_cols, self.q_state_size))
            
        self.model_eval.fit(arr_train_X, arr_train_Y, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            #self.epsilon = 1/math.exp(0.005*episode)
            #self.epsilon = 1/(1+math.exp(-(15/num_episode)*(num_episode/2-episode)))
    
    def target_update(self):
        weights = self.model_eval.get_weights()
        self.model_tar.set_weights(weights)
    
    def load(self, name):
        self.model_eval.load_weights(name)
    
    def save(self, name):
        self.model_eval.save_weights(name)
    
        
