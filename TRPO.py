import threading
import multiprocessing
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal

from helper import *
from vizdoom import *

from random import choice
from time import sleep
from time import time

def SetFromFlat(theta, scope):
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    shapes = map(var_shape, var_list)
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = np.prod(shape)
        assigns.append(v.assign(tf.reshape(theta[start:start+size],shape)))
        start += size
    return assigns

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(frame):
    s = frame[10:-10,30:-30]
    s = scipy.misc.imresize(s,[84,84])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer,sess):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.imageIn,num_outputs=16,
                kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,num_outputs=32,
                kernel_size=[4,4],stride=[2,2],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)
            
            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(hidden,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            
            self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
            self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
            self.old_act_Ps = tf.placeholder(shape=[None, a_size], name='oldpolicy', dtype=tf.float32)
            
            self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
            old_outputs = tf.reduce_sum(self.old_act_Ps * self.actions_onehot, [1])

            #Loss functions
            self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
            self.policy_loss = -tf.reduce_sum(self.responsible_outputs/old_outputs*self.advantages)

            #Get gradients from local network using local losses
            self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            #self.gradients = tf.gradients(self.loss,local_vars)
            self.pg = flatgrad(self.policy_loss, self.local_vars)

            # TRPO: KL and KL hessian vector product operator
            old_policy = tf.stop_gradient(self.policy)
            N = tf.shape(self.inputs)[0]
            Nf = tf.cast(N, tf.float32)
            self.kl = tf.reduce_sum(old_policy * tf.log(old_policy/self.policy))/Nf
            self.vector = tf.placeholder(tf.float32, shape=[None])
            grads = tf.gradients(self.kl, self.local_vars)
            shapes = map(var_shape, self.local_vars)
            start = 0
            tangents = []
            for shape in shapes:
                size = np.prod(shape)
                param = tf.reshape(self.vector[start:(start + size)], shape)
                tangents.append(param)
                start += size
            gvp = tf.reduce_sum([tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]) # gradient_vector_product = tf.reduce_sum( gradient * vector )
            self.fvp = flatgrad(gvp, self.local_vars) # hessian_vector_product = tf.grad(gradient_vector_product, local_vars)
            self.gf = GetFlat(sess, scope)
        
        with tf.variable_scope('value'):
            #value network
            self.value = slim.fully_connected(hidden,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'value')
            self.value_train = trainer.minimize(self.value_loss, var_list=local_vars)

        #compute KL and entropy
        

        self.KL = tf.reduce_sum(self.old_act_Ps * (tf.log(self.old_act_Ps) - tf.log(self.policy))) / tf.to_float(tf.shape(self.inputs)[0])
        self.entropy = tf.reduce_sum( - self.policy * tf.log(self.policy)) / tf.to_float(tf.shape(self.inputs)[0])

        summarys=[]
        for var in self.local_vars:
            summarys.append(tf.summary.histogram(var.op.name, var))
        for var in local_vars:
            summarys.append(tf.summary.histogram(var.op.name, var))
        self.summary_op = tf.summary.merge(summarys)
            

class Worker():
    def __init__(self,game,name,s_size,a_size,trainer,model_path,global_episodes,sess):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        if tf.gfile.Exists("/tmp/TRPO_train_"+str(self.number)):
            tf.gfile.DeleteRecursively("/tmp/TRPO_train_"+str(self.number))
        self.summary_writer = tf.summary.FileWriter("/tmp/TRPO_train_"+str(self.number))
        self.sess = sess
        
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer,sess)
        #self.update_local_ops = update_target_graph('global',self.name)        
        
        #The Below code is related to setting up the Doom environment
        game.set_doom_scenario_path("basic.wad") #This corresponds to the simple task we will pose our agent
        game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT)
        game.add_available_button(Button.MOVE_RIGHT)
        game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.init()
        self.actions = self.actions = np.identity(a_size,dtype=bool).tolist()
        #End Doom set-up
        self.env = game
        
    def train(self,rollout,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]
        old_act_Ps = rollout[:,6]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages,
            self.local_AC.old_act_Ps:np.vstack(old_act_Ps)}

        v_l,p_l,e_l,_ = self.sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.value_train],
            feed_dict=feed_dict) 
        
        thprev = self.local_AC.gf()
        def fisher_vector_product(p):
            feed_dict[self.local_AC.vector] = p
            return self.sess.run(self.local_AC.fvp, feed_dict=feed_dict) + 0.1 * p

        g = self.sess.run(self.local_AC.pg, feed_dict=feed_dict)
        stepdir = conjugate_gradient(fisher_vector_product, -g)
        shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
        lm = np.sqrt(shs / 0.0001) # max_kl = 0.0001
        fullstep = stepdir / lm
        neggdotstepdir = -g.dot(stepdir)

        def loss(th):
            self.sess.run(self.op, feed_dict={self.theta_pl:th})
            return self.sess.run(self.local_AC.policy_loss, feed_dict=feed_dict)

        theta, _n_backtracks, actual_improve = linesearch(loss, thprev, fullstep, -g.dot(fullstep))
        #print(np.linalg.norm(fullstep), np.linalg.norm(g),_n_backtracks, actual_improve)
        self.sess.run(self.op, feed_dict={self.theta_pl:theta})

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        kl,ent = self.sess.run([
            self.local_AC.KL,
            self.local_AC.entropy],
            feed_dict=feed_dict)

        if kl > 2*0.0001:
            self.sess.run(self.op, feed_dict={self.theta_pl:thprev})
        
        self.feed_dict = feed_dict
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout),kl,ent
        
    def work(self,max_episode_length,gamma,saver):
        episode_count = self.sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))

        self.theta_pl = tf.placeholder(dtype, [None])
        self.op = SetFromFlat(self.theta_pl, self.name)
        with self.sess.as_default(), self.sess.graph.as_default():                 
            while episode_count < max_episode_length:
                #sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                
                self.env.new_episode()
                s = self.env.get_state().screen_buffer
                episode_frames.append(s)
                s = process_frame(s)
                while self.env.is_episode_finished() == False:
                    #Take an action using probabilities from policy network output.
                    a_dist,v = self.sess.run([self.local_AC.policy,self.local_AC.value], 
                        feed_dict={self.local_AC.inputs:[s]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    r = self.env.make_action(self.actions[a]) / 100.0
                    d = self.env.is_episode_finished()
                    if d == False:
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = process_frame(s1)
                    else:
                        s1 = s
                        
                    episode_buffer.append([s,a,r,s1,d,v[0,0],a_dist[0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and d != True:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = self.sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[s]})[0,0]
                        v_l,p_l,e_l,kl,ent = self.train(episode_buffer,gamma,v1)
                        episode_buffer = []
                        #sess.run(self.update_local_ops)
                    if d == True:
                        break
                                            
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,kl,ent = self.train(episode_buffer,gamma,0.0)
                                
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 25 == 0:
                        print('\n')
                    if episode_count % 50 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images,'./TRPO_frames/image'+str(episode_count)+'.gif',
                            duration=len(images)*time_per_step,true_image=True,salience=False)
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(self.sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Policy/KL', simple_value=float(kl))
                    summary.value.add(tag='Policy/Entropy', simple_value=float(ent))
                    summary_str = self.sess.run(self.local_AC.summary_op, feed_dict=self.feed_dict)
                        
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.add_summary(summary_str, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    self.sess.run(self.increment)
                    if episode_count%1==0:
                        print('\r {} {:.6f} {:.6f}'.format(episode_count, episode_reward, kl),end=' ')
                episode_count += 1

max_episode_length = 3000
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
a_size = 3 # Agent can move Left, Right, or Fire
load_model = False
model_path = './TRPO_model'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
#Create a directory to save episode playback gifs to
if not os.path.exists('./TRPO_frames'):
    os.makedirs('./TRPO_frames')

with tf.Session() as sess:
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    #master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
    
    # Create worker classes
    worker = Worker(DoomGame(),0,s_size,a_size,trainer,model_path,global_episodes,sess)
    saver = tf.train.Saver(max_to_keep=5)


    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    worker.work(max_episode_length,gamma,saver)