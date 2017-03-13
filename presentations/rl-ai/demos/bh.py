"""
bh.py - Behavioral cloning demo.
	Author: William Guss
"""
import gym
from pyglet.window import key
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

LAMBDA = 1e-2
BATCH_SIZE=10
LEARNING_RATE = 0.001

def make_clone(sess, state_dim, act_dim):
	state = tf.placeholder(tf.float32, shape=[None, state_dim])
	expert_action = tf.placeholder(tf.float32, shape=[None, act_dim])

	print("Making model")

	with tf.variable_scope("model"):
		W1 = tf.Variable(tf.truncated_normal([state_dim, 200], stddev=0.1))
		B1 = tf.Variable(tf.truncated_normal([200], stddev=0.1))
		h = tf.nn.relu(tf.matmul(state, W1)  + B1)

		W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
		B2 = tf.Variable(tf.truncated_normal([100], stddev=0.1))
		h2 = tf.nn.relu(tf.matmul(h, W2)  + B2)


		W3 = tf.Variable(tf.truncated_normal([200, act_dim], stddev=0.1))
		B3 = tf.Variable(tf.truncated_normal([act_dim], stddev=0.1))
		output = tf.nn.tanh(tf.matmul(h, W3)   + B3)

		def feed(cur_state):
			nonlocal sess, output, state
			return sess.run(output, {state: [cur_state]})[0]


	print("Making training regime")

	with tf.variable_scope("training"):
		loss = tf.reduce_mean(tf.square(output - expert_action)) + LAMBDA*(tf.norm(W1)  + tf.norm(B1))
		train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

		def train(expert_data, batch_size=BATCH_SIZE):
			nonlocal sess, state, expert_action, loss, train_op
			exp_states = np.array(expert_data[0])
			exp_acts = np.array(expert_data[1])
			batch_ids = np.random.randint(exp_states.shape[0], size=batch_size)
			batch_x, batch_y = exp_states[batch_ids], exp_acts[batch_ids]

			error, _ = sess.run([loss, train_op],
				{state: batch_x,
				expert_action: batch_y})
			return error

	print("Initializing tensorflow computation graph.")
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	print("Clone made.")
	return feed, train


def main():
	# Expert.
	expert_states = []
	expert_actions = []
	expert_data = (expert_states, expert_actions)

	expert_a= [0]
	keypressed = False

	def key_press(k, mod):
		if k==key.LEFT:  expert_a[0] = -1.0
		if k==key.RIGHT: expert_a[0] = +1.0

	def key_release(k, mod):
		expert_a[0] = 0

	expert_env = gym.make('MountainCarContinuous-v0')
	expert_env.render()
	# from IPython import embed
	# embed()
	expert_env.env.viewer.window.on_key_press = key_press
	expert_env.env.viewer.window.on_key_release = key_release
	expert_env.env.viewer.window.set_caption('EXPERT')


	# Clone
	bh_env = gym.make('MountainCarContinuous-v0')
	bh_env.render()
	bh_env.env.viewer.window.set_caption('CLONE')
	
	feed_bh, train_bh = make_clone(tf.Session(), 2,1)


	expert_state = expert_env.reset()
	bh_state = bh_env.reset()

	i = 0
	last_reset = 0
	while True:
		i+=1
		# Train clone
		if expert_states and expert_actions:
			error = train_bh(expert_data)
			if i % 20 == 0 and i > 0:
				print("Error: {}".format(error))
				exp_states = np.array(expert_data[0])
				exp_acts = np.array(expert_data[1])

		# Handle the expert
		if abs(expert_a[0]) > 0:	
			expert_states.append(expert_state)
			expert_actions.append([expert_a[0]])
		expert_state, _, expert_done, _ = expert_env.step(expert_a)
		if expert_done or i - last_reset > 1000:
			expert_state = expert_env.reset()
			last_reset = i

		expert_env.render()


		# Handle the clone.
		bh_state, _, bh_done, _ = bh_env.step(feed_bh(bh_state))
		if bh_done or i - last_reset > 1000:
			bh_state = bh_env.reset()
			last_reset = i

		bh_env.render()







if __name__== "__main__":
	main()