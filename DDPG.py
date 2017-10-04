import itertools
import random
import gym
import numpy as np
import tensorflow as tf
from functools import partial
from helper import *

class Actor_CriticNet(object):
    def __init__(self, ob_shape, action_dim, scope, learning_rate_Q=0.001, learning_rate_A=0.0001, global_step=0):
        self.ob_shape = ob_shape
        self.action_dim = action_dim
        self.ob = tf.placeholder(tf.float32, [None, self.ob_shape], 'observation')
        self.action = tf.placeholder(tf.float32, [None, self.action_dim], 'act')
        
        #Critic neural network: Q(s,a) produces one batch of Q_values
        with tf.variable_scope(scope+'Critic_net'):
            hidden1_Q_o = tf.contrib.layers.fully_connected(self.ob, 400)
            hidden1_Q_a = tf.contrib.layers.fully_connected(self.action, 300, activation_fn=None)
            hidden2_Q_o = tf.contrib.layers.fully_connected(hidden1_Q_o, 300, activation_fn=None)
            hidden3_Q = tf.contrib.layers.fully_connected(hidden2_Q_o+hidden1_Q_a, 300)
            self.out_Q = tf.contrib.layers.fully_connected(hidden3_Q, 1, activation_fn=None)

        self.Q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+'Critic_net')
        self.target_Q = tf.placeholder(tf.float32, [None], 'target_Q')
        
        #train the NN model using semi-gradient
        self.loss_Q = tf.nn.l2_loss(self.target_Q-self.out_Q)
        self.gradient_Q = tf.gradients(self.out_Q, self.action)
        self.Optimizer_Q = tf.train.AdamOptimizer(learning_rate=learning_rate_Q)
        self.train_Q = self.Optimizer_Q.minimize(self.loss_Q, global_step=global_step, var_list=self.Q_vars)

        #self.max_Q = tf.reduce_max(self.out, axis=1)
        #self.greedy_act = tf.squeeze(tf.argmax(self.out, axis=1))
        
        #Actor neural network: A(s), using tanh to output value within (-1,1)
        with tf.variable_scope(scope+'Actor_net'):
            hidden1_A = tf.contrib.layers.fully_connected(self.ob, 400)
            hidden2_A = tf.contrib.layers.fully_connected(hidden1_A, 300)
            self.out_A = tf.contrib.layers.fully_connected(hidden2_A, self.action_dim, activation_fn=tf.nn.tanh)#, weights_initializer=tf.truncated_normal_initializer(stddev=0.1))

        self.A_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+'Actor_net')
        # chain rule: gradient_A = dQ(s,a)/da * da(s|theta)/dtheta
        self.gradient_Q_input = tf.placeholder("float", [None, self.action_dim])
        self.gradient_A = tf.gradients(self.out_A, self.A_vars, -self.gradient_Q_input)
        self.Optimizer_A = tf.train.AdamOptimizer(learning_rate=learning_rate_A)
        self.train_A  = self.Optimizer_A.apply_gradients(zip(self.gradient_A, self.A_vars))

class EMATarget(object):
    def __init__(self,from_scope,to_scope, tau=0.01):
        self.tau = tau
        self.from_vars_Q = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope+'Critic_net')
        self.from_vars_A = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope+'Actor_net')
        self.to_vars_Q = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope+'Critic_net')
        self.to_vars_A = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope+'Actor_net')

    def update_target_net(self):
        op_holder = []
        for from_var,to_var in zip(self.from_vars_Q,self.to_vars_Q):
            op_holder.append(to_var.assign((1-self.tau)*from_var+self.tau*to_var))
        for from_var,to_var in zip(self.from_vars_A,self.to_vars_A):
            op_holder.append(to_var.assign((1-self.tau)*from_var+self.tau*to_var))
        return op_holder

    def init_target_net(self):
        op_holder = []
        for from_var,to_var in zip(self.from_vars_Q,self.to_vars_Q):
            op_holder.append(to_var.assign(from_var))
        for from_var,to_var in zip(self.from_vars_A,self.to_vars_A):
            op_holder.append(to_var.assign(from_var))
        return op_holder

def main_car(total_episode=200, seed=0, gamma=0.99, animate=False):
    #make env
    env = gym.make("MountainCarContinuous-v0")
    ob_shape = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high 
    action_low = env.action_space.low 

    print('ob_shape:', ob_shape, 'action_dim', action_dim, action_high, action_low)
    #set random seed
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    ###############
    # Build Graph #
    ###############
    #Q-value Model(Linear or NN)
    global_step = tf.Variable(0, trainable=False)
    learning_rate_Q=0.001
    learning_rate_A=0.0001
    cur_Model = Actor_CriticNet(ob_shape, action_dim, 'current', learning_rate_Q, learning_rate_A, global_step)
    target_Model = Actor_CriticNet(ob_shape, action_dim, 'target', learning_rate_Q, learning_rate_A, global_step)

    buffer = Replay_buffer(memory_cap=100000)
    ounoise = OUNoise(action_dim)
    
    emaupdate = EMATarget('current', 'target', tau=0.01)
    update_target_op = emaupdate.update_target_net()
    init_target_op = emaupdate.init_target_net()
    
    batch_size = 32

    ###########
    # Run env #
    ###########
    sess = tf.Session()
    sess.__enter__()
    sess.run(tf.global_variables_initializer())
    sess.run(init_target_op)
    for i in range(total_episode):
        ounoise.reset()
        ob = env.reset()
        sess.run(init_target_op)
        if len(buffer.memory)>2*batch_size: print('\ntraining! Reward:', ep_rew)

        ac = sess.run(cur_Model.out_A, feed_dict={cur_Model.ob: ob[None,:]})[0]
        ac = ac + ounoise.noise()
        ac = np.clip(ac, action_low, action_high)
        ac = ac.reshape((-1,))

        rew = 0
        ep_rew = 0
        done = False
        observations =[]
        actions = []
        rewards = []
        targets = []
        for t in itertools.count():
            if animate and i%10 == 0:
                env.render()
            #run env
            ob_next, rew, done, info = env.step(ac)
            ep_rew += rew
            buffer.add_framework(ob, ac, ob_next, rew, done)

            print('\r\t episode:{}, step:{} learning_rate_Q:{:6.4f} learning_rate_A:{:6.4f}'.format(i, t, learning_rate_Q, learning_rate_A), end='')
            if done:
                #if i%10==0: print('\t seed:{}, episode:{}, step:{}, rewards:{:8f}, learning_rate:{:8.4f}'.format(seed, i, t, ep_rew, learning_rate))
                break
            
            #action with OUNoise
            ac_next = sess.run(cur_Model.out_A, feed_dict={cur_Model.ob: ob_next[None,:]})[0]
            ac_next = ac_next + ounoise.noise()
            ac_next = np.clip(ac_next, action_low, action_high)
            ac_next = ac_next.reshape((-1,))

            #train the Q_net with replay buffer
            if len(buffer.memory)>2*batch_size: 
                ob_batch, ac_batch, next_ob_batch, rew_batch, done_batch = buffer.get_batch(batch_size)
                # Q_net target: r + gamma*Q'(s', a'(s'|theta')) (expected Sarsa)
                next_a = sess.run(target_Model.out_A, feed_dict={target_Model.ob:next_ob_batch})
                next_Q = np.squeeze(sess.run(target_Model.out_Q, feed_dict={target_Model.ob:next_ob_batch, target_Model.action:next_a}))
                target_Q = rew_batch + gamma*next_Q*(1-done_batch)
                # Update current Critic net
                _, gradient_Q = sess.run([cur_Model.train_Q, cur_Model.gradient_Q], feed_dict={cur_Model.ob:ob_batch, 
                                                                                               cur_Model.action:ac_batch, 
                                                                                               cur_Model.target_Q:target_Q})
                
                # Update current Actor net: dQ(s,a(s|theta))/dtheta
                cur_a = sess.run(cur_Model.out_A, feed_dict={cur_Model.ob:ob_batch})
                sess.run(cur_Model.train_A, feed_dict={cur_Model.ob:ob_batch,  
                                                       cur_Model.gradient_Q_input:gradient_Q[0]})
                # Update target net
                if i % 100 == 0:
                    sess.run(init_target_op)
                
            ob = ob_next
            ac = ac_next

if __name__ == "__main__":
    main_car()