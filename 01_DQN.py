# Import modules
import tensorflow as tf
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import cv2
from collections import deque
import os

# Import game
import sys
sys.path.append("DQN_GAMES/")

import Parameters
game = Parameters.game

# Game Information
algorithm = 'DQN'
game_name = game.ReturnName()

# Get parameters
Num_action = game.Return_Num_Action()

# Initial parameters
Num_Exploration = Parameters.Num_start_training
Num_Training    = Parameters.Num_training
Num_Testing     = Parameters.Num_test

lr = Parameters.Learning_rate
gamma = Parameters.Gamma

first_epsilon = Parameters.Epsilon
final_epsilon = Parameters.Final_epsilon

Num_plot_step = Parameters.Num_plot_step

# date - hour - minute - second of training time
date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

train_mode = Parameters.Train_mode
load_model = Parameters.Load_model
load_path = Parameters.Load_path
save_path = 'saved_networks/' + game_name + '/' + date_time + '_' + algorithm

# parameters for skipping and stacking
Num_skipping = Parameters.Num_skipFrame
Num_stacking = Parameters.Num_stackFrame

# Parameter for Experience Replay
Num_replay_memory = Parameters.Num_replay_memory
Num_batch = Parameters.Num_batch

# Parameter for Target Network
Num_update_target = Parameters.Num_update

# Parameters for network
img_size = 80
Num_colorChannel = Parameters.Num_colorChannel

class Model:
	def __init__(self, network_name):
		# Input
		self.x = tf.placeholder(tf.float32, shape = [None, img_size, img_size,
											         Num_stacking * Num_colorChannel])
		x_normalize = (self.x-(255.0/2)) / (255.0/2)

		with tf.variable_scope(network_name):
			conv1 = tf.layers.conv2d(inputs=x_normalize, filters=32, activation=tf.nn.relu, kernel_size=[8,8], strides=[4,4], padding="SAME")
			conv2 = tf.layers.conv2d(inputs=conv1, filters=64, activation=tf.nn.relu, kernel_size=[4,4], strides=[2,2],padding="SAME")
			conv3 = tf.layers.conv2d(inputs=conv2, filters=64, activation=tf.nn.relu, kernel_size=[3,3], strides=[1,1],padding="SAME")
			flat = tf.layers.flatten(conv3)
			fc1 = tf.layers.dense(flat,512,activation=tf.nn.relu)
			self.Q_Out = tf.layers.dense(fc1, Num_action, activation=None)

		# Loss function and Train
		self.action_onehot = tf.placeholder(tf.float32, shape = [None, Num_action])
		self.y_target = tf.placeholder(tf.float32, shape = [None])

		self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, network_name)

		y_pred = tf.reduce_sum(tf.multiply(self.Q_Out, self.action_onehot), axis=1)
		self.Loss = tf.reduce_mean(tf.square(y_pred - self.y_target))
		self.train_step = tf.train.AdamOptimizer(learning_rate = lr, epsilon = 1e-02).minimize(self.Loss, var_list=self.trainable_var)

class DQN:
	def __init__(self):
		self.progress = ''
		self.train_mode = train_mode

		# initialize step, score and episode
		self.step = 1
		self.score = 0
		self.episode = 0

		self.epsilon = first_epsilon

		self.replay_memory = deque(maxlen=Num_replay_memory)
		self.state_set = deque(maxlen=Num_skipping*Num_stacking)

		# Variables for tensorboard
		self.loss_list = []
		self.score_list = []
		self.maxQ_list = []

		# Initialize Network
		self.model = Model('network')
		self.target_model = Model('target')
		self.sess, self.saver = self.init_sess()
		self.Summary, self.Merge = self.Make_Summary()

	def init_sess(self):
		# Initialize variables
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		sess = tf.InteractiveSession(config=config)

		# Make folder for save data
		os.makedirs(save_path)

		init = tf.global_variables_initializer()
		sess.run(init)

		# Load the file if the saved file exists
		saver = tf.train.Saver()

		if load_model:
			# Restore variables from disk.
			saver.restore(sess, load_path + "/model")
			print("Model restored.")

		return sess, saver

	def get_progress(self):
		if self.step <= Num_Exploration and self.train_mode:
			self.progress = 'Exploring'
		elif self.step <= Num_Exploration + Num_Training and self.train_mode:
			self.progress = 'Training'
		elif self.step <= Num_Exploration + Num_Training + Num_Testing:
			self.progress = 'Testing'
		else:
			self.progress = 'Finished'

	def init_state(self, game_state):
		action = np.zeros([Num_action])
		state, _, _ = game_state.frame_step(action)
		state = self.reshape_input(state)

		for i in range(Num_skipping * Num_stacking):
			self.state_set.append(state)

		return state

	# Resize and make input as grayscale
	def reshape_input(self, state):
		state_out = cv2.resize(state, (img_size, img_size))
		if Num_colorChannel == 1:
			state_out = cv2.cvtColor(state_out, cv2.COLOR_BGR2GRAY)
			state_out = np.reshape(state_out, (img_size, img_size, 1))
			
		return state_out

	def skip_and_stack_frame(self, state):
		self.state_set.append(state)

		state_in = np.zeros((img_size, img_size, Num_colorChannel * Num_stacking))

		# Stack the frame according to the number of skipping frame
		for stack_frame in range(Num_stacking):
			state_in[:,:, Num_colorChannel * stack_frame : Num_colorChannel * (stack_frame+1)] = self.state_set[-1 - (Num_skipping * stack_frame)]

		state_in = np.uint8(state_in)
		return state_in

	# Tensorboard  
	def Make_Summary(self):
		self.summary_loss   = tf.placeholder(dtype=tf.float32)
		self.summary_reward = tf.placeholder(dtype=tf.float32)
		self.summary_maxQ   = tf.placeholder(dtype=tf.float32)
		tf.summary.scalar("loss", self.summary_loss)
		tf.summary.scalar("reward", self.summary_reward)
		tf.summary.scalar("maxQ", self.summary_maxQ)
		Summary = tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph)
		Merge = tf.summary.merge_all()

		return Summary, Merge

	def Write_Summray(self, reward, loss, maxQ, step):
		self.Summary.add_summary(
			self.sess.run(self.Merge, feed_dict={self.summary_loss: loss, 
												 self.summary_reward: reward,
												 self.summary_maxQ: maxQ}), step)

	def select_action(self, state):
		action = np.zeros([Num_action])
		action_index = 0

		# Choose action
		if random.random() < self.epsilon:
			# Choose random action
			action_index = random.randint(0, Num_action-1)
			action[action_index] = 1
		else:
			# Choose greedy action
			Q_value = self.model.Q_Out.eval(feed_dict={self.model.x: [state]})
			action_index = np.argmax(Q_value)
			action[action_index] = 1
			self.maxQ_list.append(np.max(Q_value))

		return action

	def experience_replay(self, state, action, reward, next_state, done):
		self.replay_memory.append([state, action, reward, next_state, done])

	def update_target(self):
		for i in range(len(self.model.trainable_var)):
			self.sess.run(self.target_model.trainable_var[i].assign(self.model.trainable_var[i]))

	def train(self):
		# Decrease epsilon while training
		if self.epsilon > final_epsilon:
			self.epsilon -= first_epsilon/Num_Training

		# Select minibatch
		minibatch =  random.sample(self.replay_memory, Num_batch)

		# Save the each batch data
		state_batch      = [batch[0] for batch in minibatch]
		action_batch     = [batch[1] for batch in minibatch]
		reward_batch     = [batch[2] for batch in minibatch]
		next_state_batch = [batch[3] for batch in minibatch]
		done_batch       = [batch[4] for batch in minibatch]

		# Get y_prediction
		y_batch = []
		Q_batch = self.target_model.Q_Out.eval(feed_dict = {self.target_model.x: next_state_batch})

		# Get target values
		for i in range(len(minibatch)):
			if done_batch[i] == True:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + gamma * np.max(Q_batch[i]))

		_, loss = self.sess.run([self.model.train_step, self.model.Loss], 
		                         feed_dict = {self.model.action_onehot: action_batch,
											  self.model.y_target: y_batch,
											  self.model.x: state_batch})

		self.loss_list.append(loss)

	def save_model(self):
		# Save the variables to disk.
		if self.step == Num_Exploration + Num_Training:
			save_path_ = self.saver.save(self.sess, save_path + "/model/model")
			print("Model saved in file: %s" % save_path_)

	def plot(self):
		if self.step % Num_plot_step == 0 and len(self.score_list)>5:
			self.Write_Summray(np.mean(self.score_list), np.mean(self.loss_list), np.mean(self.maxQ_list), self.step)
			self.score_list = []
			self.loss_list = []
			self.maxQ_list = []

	def is_done(self, game_state):
		# Show Progress
		print('Step: {} / Episode: {} / Progress: {} / Loss: {:.4f} / Epsilon: {:.4f} / Score: {:.3f}'.format(
			   self.step, self.episode, self.progress, np.mean(self.loss_list), self.epsilon, self.score))

		self.score_list.append(self.score)

		self.episode += 1
		self.score = 0

		# If game is finished, initialize the state
		state = self.init_state(game_state)
		state = self.skip_and_stack_frame(state)

		return state

if __name__ == '__main__':
	agent = DQN()

	# Define game state
	game_state = game.GameState()

	# Initialization
	state = agent.init_state(game_state)
	state = agent.skip_and_stack_frame(state)

	while True:
		agent.get_progress()
		
		# Select action
		action = agent.select_action(state)

		# Take action and get info. for update
		next_state, reward, done = game_state.frame_step(action)
		next_state = agent.reshape_input(next_state)
		next_state = agent.skip_and_stack_frame(next_state)

		# Experience Replay
		agent.experience_replay(state, action, reward, next_state, done)

		# Training!
		if agent.progress=='Training':
			# Update target network
			if agent.step % Num_update_target == 0:
				agent.update_target()
			agent.train()
			agent.save_model()

		# Testing 
		if agent.progress=='Testing':
			agent.epsilon = 0.

		# Update former info.
		state = next_state
		agent.score += reward
		agent.step += 1

		# If game is over (done=True)
		if done:
			state = agent.is_done(game_state)

		agent.plot()

		# Finished!
		if agent.progress=='Finished':
			print('Finished!')
			break
