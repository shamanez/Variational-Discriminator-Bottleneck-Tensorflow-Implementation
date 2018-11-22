from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim

from functools import partial
import pdb



def generator(z, dim=64, reuse=True):

    '''
    simple generator to generate mnist type data
    Inputs - Noise vector and dim describing the paramters of a a hidden layer

    '''

    with tf.variable_scope('generator', reuse=reuse):
        y = tf.contrib.layers.fully_connected(inputs=z, num_outputs=1024,activation_fn=tf.nn.relu)
        y = tf.contrib.layers.fully_connected(inputs=y, num_outputs=7 * 7 * dim * 2,activation_fn=tf.nn.relu)
        y = tf.reshape(y, [-1, 7, 7, dim * 2])
        y = tf.contrib.layers.conv2d_transpose(inputs=y, num_outputs=dim * 2, kernel_size=5, stride=2,activation_fn=tf.nn.relu)
        img = tf.tanh(tf.contrib.layers.conv2d_transpose(inputs=y, num_outputs=1, kernel_size=5, stride=2,activation_fn=None))
        return img



def discriminator_wgan_gp(img, dim=64, reuse=True,gen_train=False,bottleneck_dim=512):

    '''
    Discriminator modified with Variational Discriminator Bottleneck
    Inputs - Noise vector and dim describing the paramters of a a hidden layer.
    bottleneck_dim describes how many dimentions for bottleneck layer
    
    '''

    with tf.variable_scope('discriminator', reuse=reuse):
        y = tf.contrib.layers.conv2d(inputs=img, num_outputs=1, kernel_size=5, stride=2,activation_fn=tf.nn.relu)
        y = tf.contrib.layers.conv2d(inputs=y, num_outputs=dim, kernel_size=5, stride=2,activation_fn=tf.nn.relu)
        y = tf.contrib.layers.flatten(y)
        y = tf.contrib.layers.fully_connected(inputs=y, num_outputs=bottleneck_dim*2,activation_fn=tf.nn.relu)
        params=y.shape[-1]//2

        mus=y[:,:params] #first 512 is mus
        sigmas=y[:,params:] #Second 512 is stds w.r.t the dimentions
        
        #This is importants We call this reparameterization trick trick 
        #We sample w.r.t to fixed gassian distribution
        #Here We have two options how to sample either taking the mean or sample values when training the generator
        #and the discriminator 

        #Please refer the last section of the part 4 in the VDB paper
        if not gen_train:
            eps=tf.keras.backend.random_normal(shape=(32,512),mean=0,stddev=1)
            bottle_out=mus+ sigmas*eps
        else:
            bottle_out=mus
     
        bottle_out=tf.nn.leaky_relu(bottle_out)    
        logit = tf.contrib.layers.fully_connected(inputs=bottle_out, num_outputs=1,activation_fn=None)
 

        return logit,mus,sigmas


