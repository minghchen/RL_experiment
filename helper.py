import numpy as np
import numpy.random as nr
import random
import tensorflow as tf
#import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
from collections import deque
import tensorflow.contrib.slim as slim
import tensorflow as tf 

########
# DDPG #
########
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

class OUNoise(object):
    """docstring for OUNoise"""
    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

########
# TRPO #
########
dtype = tf.float32

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
'''
class SetFromFlat(object):

    def __init__(self, session, scope):
        self.session = session
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        assigns = []
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(tf.assign(v,tf.reshape(theta[start:start +size],shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})
'''

class GetFlat(object):

    def __init__(self, session, scope):
        self.session = session
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)

    def __call__(self):
        return self.op.eval(session=self.session)

# use to get all variable shapes
def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

def numel(x):
    return np.prod(var_shape(x))

def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], 0)

# liner search on the grendient direction
def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(0.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        #print(np.linalg.norm(xnew),np.linalg.norm(fullstep),stepfrac,actual_improve,expected_improve)
        if ratio > accept_ratio and actual_improve > 0:
            return xnew, _n_backtracks, actual_improve
    return x, _n_backtracks, actual_improve

# CG: conjugate gradient
def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x

########
# DOOM #
########
#This is a simple function to reshape our game frames.
def processState(state1):
    return np.reshape(state1,[21168])
    
#These functions allows us to update the parameters of our target network with those of the primary network.
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
    total_vars = len(tf.trainable_variables())
    a = tf.trainable_variables()[0].eval(session=sess)
    b = tf.trainable_variables()[total_vars//2].eval(session=sess)
    if a.all() == b.all():
        print("Target Set Success")
    else:
        print("Target Set Failed")
        
#Record performance metrics and episode logs for the Control Center.
def saveToCenter(i,rList,jList,bufferArray,summaryLength,h_size,sess,mainQN,time_per_step):
    with open('./Center/log.csv', 'a') as myfile:
        state_display = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        imagesS = []
        for idx,z in enumerate(np.vstack(bufferArray[:,0])):
            img,state_display = sess.run([mainQN.salience,mainQN.rnn_state],\
                feed_dict={mainQN.scalarInput:np.reshape(bufferArray[idx,0],[1,21168])/255.0,\
                mainQN.trainLength:1,mainQN.state_in:state_display,mainQN.batch_size:1})
            imagesS.append(img)
        imagesS = (imagesS - np.min(imagesS))/(np.max(imagesS) - np.min(imagesS))
        imagesS = np.vstack(imagesS)
        imagesS = np.resize(imagesS,[len(imagesS),84,84,3])
        luminance = np.max(imagesS,3)
        imagesS = np.multiply(np.ones([len(imagesS),84,84,3]),np.reshape(luminance,[len(imagesS),84,84,1]))
        make_gif(np.ones([len(imagesS),84,84,3]),'./Center/frames/sal'+str(i)+'.gif',duration=len(imagesS)*time_per_step,true_image=False,salience=True,salIMGS=luminance)

        images = zip(bufferArray[:,0])
        images.append(bufferArray[-1,3])
        images = np.vstack(images)
        images = np.resize(images,[len(images),84,84,3])
        make_gif(images,'./Center/frames/image'+str(i)+'.gif',duration=len(images)*time_per_step,true_image=True,salience=False)

        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow([i,np.mean(jList[-100:]),np.mean(rList[-summaryLength:]),'./frames/image'+str(i)+'.gif','./frames/log'+str(i)+'.csv','./frames/sal'+str(i)+'.gif'])
        myfile.close()
    with open('./Center/frames/log'+str(i)+'.csv','w') as myfile:
        state_train = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["ACTION","REWARD","A0","A1",'A2','A3','V'])
        a, v = sess.run([mainQN.Advantage,mainQN.Value],\
            feed_dict={mainQN.scalarInput:np.vstack(bufferArray[:,0])/255.0,mainQN.trainLength:len(bufferArray),mainQN.state_in:state_train,mainQN.batch_size:1})
        wr.writerows(zip(bufferArray[:,1],bufferArray[:,2],a[:,0],a[:,1],a[:,2],a[:,3],v[:,0]))
    
#This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
  import moviepy.editor as mpy
  
  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)
  
  def make_mask(t):
    try:
      x = salIMGS[int(len(salIMGS)/duration*t)]
    except:
      x = salIMGS[-1]
    return x

  clip = mpy.VideoClip(make_frame, duration=duration)
  if salience == True:
    mask = mpy.VideoClip(make_mask, ismask=True,duration= duration)
    clipB = clip.set_mask(mask)
    clipB = clip.set_opacity(0)
    mask = mask.set_opacity(0.1)
    mask.write_gif(fname, fps = len(images) / duration,verbose=False)
    #clipB.write_gif(fname, fps = len(images) / duration,verbose=False)
  else:
    clip.write_gif(fname, fps = len(images) / duration,verbose=False)