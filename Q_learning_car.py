import itertools
import random
from collections import deque
import gym
import numpy as np
import tensorflow as tf

class NNQValue(object):
    def __init__(self, ob_shape, action_num, learning_rate=0.01, global_step=0):
        self.input_shape = ob_shape
        self.action_num = action_num
        #neuro network model with one hidden layer
        self.input = tf.placeholder(tf.float32, [None, self.input_shape], 'observation')
        hidden1 = tf.contrib.layers.fully_connected(self.input, 64)
        hidden2 = tf.contrib.layers.fully_connected(hidden1, 64)
        self.out = tf.contrib.layers.fully_connected(hidden2, action_num, activation_fn=None)

        self.targetQ = tf.placeholder(tf.float32, [None], 'targetQ')
        self.action = tf.placeholder(tf.int32, [None], 'act')
        self.Q_a = tf.reduce_sum(self.out*tf.one_hot(self.action, action_num),axis=1)
        #train the NN model using semi-gradient
        self.loss = tf.nn.l2_loss(self.targetQ-self.Q_a)
        self.Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train = self.Optimizer.minimize(self.loss, global_step=global_step)

        self.max_Q = tf.reduce_max(self.out, axis=1)
        self.greedy_act = tf.squeeze(tf.argmax(self.out, axis=1))

def epsilon_decay(step=0, eps_min=0.05, eps_max=1.0, eps_decay_steps = 100):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    return epsilon

class Replay_buffer(object):
    def __init__(self, memory_cap=10000):
        self.memory_cap = memory_cap
        self.memory = deque(maxlen=memory_cap)
    def add_framework(self, ob, ac, next_ob, rew, done):
        self.memory.append([ob, ac, next_ob, rew, done])
    def get_batch(self, batch_size=256):
        if len(self.memory) < 2*batch_size:
            raise Exception('no enough memory')
        idx = np.random.permutation(len(self.memory))[:batch_size]
        extract_mem = lambda k : np.array([self.memory[i][k] for i in idx])
        ob_batch = extract_mem(0)
        ac_batch = extract_mem(1)
        next_ob_batch = extract_mem(2)
        rew_batch = extract_mem(3)
        done_batch = extract_mem(4)
        return ob_batch, ac_batch, next_ob_batch, rew_batch, done_batch

def main_car(total_episode=1000, seed=0, gamma=0.99, epsilon=0.01, step_size=0.01, animate=False):
    #make env
    env = gym.make("MountainCar-v0")
    ob_shape = env.observation_space.shape[0]
    action_num = env.action_space.n 

    #set random seed
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    #Q-value Model(Linear or NN)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.01, global_step=global_step, decay_steps=350000, decay_rate=0.01)
    NN_Q = NNQValue(ob_shape, action_num, learning_rate, global_step)

    buffer = Replay_buffer()
    train_begin = 10

    sess = tf.Session()
    sess.__enter__()
    sess.run(tf.global_variables_initializer())
    for i in range(total_episode):
        #initialize the env with epsilon_greedy action
        ob = env.reset()
        epsilon = epsilon_decay(i)
        if random.random() < epsilon or i < train_begin:
            ac = env.action_space.sample()
        else:
            ac = sess.run(NN_Q.greedy_act, feed_dict={NN_Q.input: ob[None,:]})

        rew = 0
        ep_rew = 0
        done = False
        observations =[]
        actions = []
        rewards = []
        targets = []
        for t in itertools.count():
            if animate and i%100 == 0:
                env.render()
            #run env
            ob_next, rew, done, info = env.step(ac)
            ep_rew += rew
            buffer.add_framework(ob, ac, ob_next, rew, done)
            #observations.append(ob)
            #actions.append(ac)
            #rewards.append(rew)

            print('\r\t episode:{}, step:{}, epsilon:{:8.4f}, learning_rate:{:8.4f}'.format(i, t, epsilon, learning_rate.eval()), end='')
            if done:
                if i%10==0: print('\t seed:{}, episode:{}, step:{}, rewards:{}, epsilon:{:8.4f}, learning_rate:{:8.4f}'.format(seed, i, t, ep_rew, epsilon, learning_rate.eval()))
                """
                ###Sarsa
                target_Q = rew
                sess.run(NN_Q.train, feed_dict={NN_Q.input:ob[None,:], NN_Q.action:[ac], NN_Q.targetQ:[target_Q]})
                #episodic off-line training with TD(0)
                observations = np.array(observations)
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_Q = sess.run(tf.squeeze(NN_Q.Q_a), feed_dict={NN_Q.input: observations[1:,:], NN_Q.action:actions[1:]})
                targets = list(rewards[:-1] + gamma*next_Q)
                targets.append(rew)
                #train the NN model using semi-gradient
                sess.run(NN_Q.train, feed_dict={NN_Q.input:observations, NN_Q.action:actions, 
                                                    NN_Q.targetQ:np.array(targets)})
                """

                break
            #epsilon_greedy action
            if random.random() < epsilon or i < train_begin:
                ac_next = env.action_space.sample()
            else:
                ac_next = sess.run(NN_Q.greedy_act, feed_dict={NN_Q.input: ob_next[None,:]})
            #train the Q_net with replay buffer
            if i > train_begin: 
                ob_batch, ac_batch, next_ob_batch, rew_batch, done_batch = buffer.get_batch()
                next_Q = sess.run(NN_Q.max_Q, feed_dict={NN_Q.input: next_ob_batch})
                target_Q = rew_batch + gamma*next_Q*(1-done_batch)
                sess.run(NN_Q.train, feed_dict={NN_Q.input:ob_batch, NN_Q.action:ac_batch, NN_Q.targetQ:target_Q})
            ob = ob_next
            ac = ac_next

def main_car1(d):
    return main_car(**d)

if __name__ == "__main__":
    if 1:
        main_car(total_episode=30000, seed=0, epsilon=0.01, step_size=0.01, animate=True)
    if 0:
        params = [
            dict(total_episode=1000, seed=0, epsilon=0.01, step_size=0.01),
            dict(total_episode=1000, seed=1, epsilon=0.01, step_size=0.01),
            dict(total_episode=1000, seed=0, epsilon=0.05, step_size=0.01),
            dict(total_episode=1000, seed=1, epsilon=0.05, step_size=0.01),
            dict(total_episode=1000, seed=0, epsilon=0.01, step_size=0.05),
            dict(total_episode=1000, seed=1, epsilon=0.01, step_size=0.05),
        ]
        import multiprocessing
        p = multiprocessing.Pool()
        p.map(main_car1, params)