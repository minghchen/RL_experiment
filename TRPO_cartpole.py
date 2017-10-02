import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal

from helper import *

def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b

def fancy_slice_2d(X, inds0, inds1):
    """
    Like numpy's X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y]. 
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1    
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the 
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)

def pathlength(path):
    return len(path["reward"])

class LinearValueFunction(object):
    coef = None
    def fit(self, X, y):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)
    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)
    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

class NnValueFunction(object):
    # YOUR CODE HERE
    def __init__(self, ob_dim, n_epochs=10, stepsize=1e-3):
        self.ob_dim = ob_dim
        self.n_epochs = n_epochs
        self.stepsize = stepsize

        with tf.variable_scope("Value_function", reuse=False):
            self.shape = 2*ob_dim + 1
            self.inputs = tf.placeholder(tf.float32, [None, self.shape], name="inputs")
            self.vtargs = tf.placeholder(tf.float32, [None], name="vtargs")
            h1 = lrelu(dense(self.inputs, 32, "hidden1", normc_initializer(std=1.0)))
            h2 = lrelu(dense(h1, 16, "hidden2", normc_initializer(std=0.1)))
            out = dense(h2, 1, "Value", normc_initializer(std=0.1))
            self.out = tf.reshape(out, [-1])
        self.loss = tf.nn.l2_loss(self.out-self.vtargs)
        self.Optimizer = tf.train.AdamOptimizer(learning_rate=self.stepsize)
        self.train = self.Optimizer.minimize(self.loss)

    def predict(self, X):
        self.sess = tf.get_default_session()
        return self.sess.run(self.out, feed_dict={self.inputs:self.preproc(X)})

    def fit(self, ob_no, vtarg_n):
        for t in range(self.n_epochs):
            self.sess.run(self.train, feed_dict={self.inputs:self.preproc(ob_no), self.vtargs:vtarg_n})

    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def main_cartpole(n_iter=500, gamma=1.0, min_timesteps_per_batch=1000, stepsize=1e-2, animate=False, logdir=None):
    env = gym.make("CartPole-v0")
    ob_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    logz.configure_output_dir(logdir)
    vf = LinearValueFunction()
    seed = 0
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    #########
    # Model #
    #########
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) # advantage function estimate
    
    sy_oldlogits_na = tf.placeholder(shape=[None, num_actions], name='oldlogits', dtype=tf.float32) # logits BEFORE update (just used for KL diagnostic)

    with tf.variable_scope('actor_net'):
        sy_h1 = lrelu(dense(sy_ob_no, 32, "h1", weight_init=normc_initializer(1.0))) # hidden layer
        sy_logits_na = dense(sy_h1, num_actions, "final", weight_init=normc_initializer(0.05)) # "logits", describing probability distribution of final layer
        # we use a small initialization for the last layer, so the initial policy has maximal entropy
        sy_logp_na = tf.nn.log_softmax(sy_logits_na) # logprobability of actions
        sy_sampled_ac = categorical_sample_logits(sy_logits_na)[0] # sampled actions, used for defining the policy (NOT computing the policy gradient)
        sy_n = tf.shape(sy_ob_no)[0]
        sy_logprob_n = fancy_slice_2d(sy_logp_na, tf.range(sy_n), sy_ac_n) # log-prob of actions taken -- used for policy gradient calculation
    
    # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC PURPOSES >>>>
    sy_oldlogp_na = tf.nn.log_softmax(sy_oldlogits_na)
    sy_oldp_na = tf.exp(sy_oldlogp_na) 
    sy_kl = tf.reduce_sum(sy_oldp_na * (sy_oldlogp_na - sy_logp_na)) / tf.to_float(sy_n)
    sy_p_na = tf.exp(sy_logp_na)
    sy_ent = tf.reduce_sum( - sy_p_na * sy_logp_na) / tf.to_float(sy_n)
    # <<<<<<<<<<<<<

    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

    #Get gradients from local network using local losses
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_net')
    #self.gradients = tf.gradients(self.loss,local_vars)
    pg = flatgrad(sy_surr, local_vars)

    # TRPO: KL and KL hessian vector product operator
    policy = tf.nn.softmax(sy_logits_na)
    old_policy = tf.stop_gradient(policy)
    N = tf.shape(sy_ob_no)[0]
    Nf = tf.cast(N, tf.float32)
    kl_tr = tf.reduce_sum(old_policy * tf.log(old_policy/policy))/Nf
    vector = tf.placeholder(tf.float32, name='vector', shape=[None])
    grads = tf.gradients(kl_tr, local_vars)
    shapes = map(var_shape, local_vars)
    start = 0
    tangents = []
    for shape in shapes:
        size = np.prod(shape)
        param = tf.reshape(vector[start:(start + size)], shape)
        tangents.append(param)
        start += size
    gvp = tf.reduce_sum([tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]) # gradient_vector_product = tf.reduce_sum( gradient * vector )
    fvp = flatgrad(gvp, local_vars) # hessian_vector_product = tf.grad(gradient_vector_product, local_vars)

    
    ###########
    #Begin env#
    ###########
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    # use single thread. on such a small problem, multithreading gives you a slowdown
    # this way, we can better use multiple cores for different experiments
    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    total_timesteps = 0

    gf = GetFlat(sess, 'actor_net')
    theta_pl = tf.placeholder(dtype, [None])
    op = SetFromFlat(theta_pl, 'actor_net')
    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break                    
            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        feed_dict = {sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize}
        oldlogits_na = sess.run(sy_logits_na, feed_dict=feed_dict)
        
        # Policy update
        #TRPO
        thprev = gf()
        def fisher_vector_product(p):
            feed_dict[vector] = p
            return sess.run(fvp, feed_dict=feed_dict) + 0.1 * p

        g = sess.run(pg, feed_dict=feed_dict)
        stepdir = conjugate_gradient(fisher_vector_product, -g)
        shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
        lm = np.sqrt(shs / 0.01) # max_kl = 0.01
        fullstep = stepdir / lm
        neggdotstepdir = -g.dot(stepdir)

        def loss(th):
            sess.run(op, feed_dict={theta_pl:th})
            return sess.run(sy_surr, feed_dict=feed_dict)

        theta, _n_backtracks, actual_improve = linesearch(loss, thprev, fullstep, -g.dot(fullstep))
        #print(np.linalg.norm(fullstep), np.linalg.norm(g),_n_backtracks, actual_improve)
        sess.run(op, feed_dict={theta_pl:theta})

        
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no, sy_oldlogits_na:oldlogits_na})

        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()

if __name__ == "__main__":
    main_cartpole(logdir='./TRPO_cartpole') # when you want to start collecting results, set the logdir