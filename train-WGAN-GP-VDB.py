'''
This is the Tensorflow Implementation - VARIATIONAL DISCRIMINATOR BOTTLENECK: IMPROVING IMITATION LEARNING, INVERSE RL, AND GANS BY
CONSTRAINING INFORMATION FLOW
'''

#This implemetation is heavily influenced by following pytoch implementation 
#Git-Hub Repo- https://github.com/akanimax/Variational_Discriminator_Bottleneck


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import utils
import traceback
import numpy as np
import tensorflow as tf
import data_mnist as data
import models as models
import vdb_losses as losses

import pdb



def main(epoch,batch_size,lr,z_dim,bottle_dim,i_c,alpha,n_critic,gpu_id,data_pool):


    with tf.device('/gpu:%d' % gpu_id): #Placing the ops under devices

        generator = models.generator  #Generator Object
        discriminator = models.discriminator_wgan_gp #Discriminator Object

        # inputs Placeholders
        real = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        z = tf.placeholder(tf.float32, shape=[None, z_dim])

        # generate fake data with the generator
        fake = generator(z, reuse=False)

        # Obtaining scores , means and stds for real and fake data from the discriminator
        r_logit,r_mus,r_sigmas = discriminator(real, reuse=False,gen_train=False,bottleneck_dim=bottle_dim,batch_size=batch_size)
        f_logit,f_mus,f_sigmas = discriminator(fake,gen_train=False,bottleneck_dim=bottle_dim,batch_size=batch_size)

        
        #Obtaining wasserstein loss and gradient penalty losses to train the discriminator
        wasserstein_d=losses.wgan_loss(r_logit,f_logit)
        gp = losses.gradient_penalty(real, fake, discriminator,batch_size=batch_size)


        #We obtain the bottleneck loss in the discriminator 
        #Inputs to this function are bottleneck layer mus and stds for both real and fake data. i_c is the
        #the information constriant or upperbound. This is an important paramters 
        bottleneck_loss=losses._bottleneck_loss(real_mus=r_mus, fake_mus=f_mus,\
            real_sigmas=r_sigmas,fake_sigmas=f_sigmas,i_c=i_c)



        #This used in lagrangian multiplier optimization. This is paramters also get updated adaptivly. 
        #To read more about duel gradient desenet in deep learning please read - https://medium.com/@jonathan_hui/rl-dual-gradient-descent-fac524c1f049
        #Initialize with the zero

        beta=tf.Variable(tf.zeros([]), name="beta")



        #Combined both losses (10 is the default hyper paramters given by the paper 
        # - https://arxiv.org/pdf/1704.00028.pdf )
        d_loss = -wasserstein_d + gp * 10.0 + beta*bottleneck_loss

        #We said b also should adaptively get updated. Here we maximize the beta paramters with follwoing function
        #Please refer to the VDB paper's equation (9) understand more about the update 
        beta_new=tf.maximum(0.0,beta+alpha*bottleneck_loss)

        #This is the main difference from the pytoch implementation. In tensorlfow we have a static graph. S
        # to update the beta with above menitoned function we have to use tf.assign()
        assign_op=tf.assign(beta,beta_new)  #beta.assign(beta_new)


        #This is the generator loss 
        #As described in the paper we have a simple loss to the generator which uses mean scores from 
        #the generated samples 
        f_logit_gen,f_mus_gen,f_sigmas_gen = discriminator(fake,gen_train=True,bottleneck_dim=bottle_dim,batch_size=batch_size)
        g_loss = -tf.reduce_mean(f_logit_gen)


        #Assigning two optimizers to train both Generator and the Discriminator
        d_var = tf.trainable_variables('discriminator')
        g_var = tf.trainable_variables('generator')

        d_step = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(d_loss, var_list=d_var)
        g_step = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(g_loss, var_list=g_var)


   


    # Tensorbored summaries for  plot losses
    wd=wasserstein_d
    d_summary = utils.summary({wd: 'wd', gp: 'gp'})
    g_summary = utils.summary({g_loss: 'g_loss'})
    beta_summary = utils.b_summary(beta)
    #beta_summary = utils.summary({beta: 'beta'})    

    
    #sess= tf.Session()

    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    with tf.Session(config=config) as sess:
        # iteration counter
        it_cnt, update_cnt = utils.counter()
        # saver
        saver = tf.train.Saver(max_to_keep=5) #Use to save both generator and discriminator paramters
        # summary writer
        summary_writer = tf.summary.FileWriter('./summaries/mnist_wgan_gp', sess.graph)



        ''' Checking for previuosly trained checkpints'''
        ckpt_dir = './checkpoints/mnist_wgan_gp'
        utils.mkdir(ckpt_dir + '/')
        if not utils.load_checkpoint(ckpt_dir, sess):
            sess.run(tf.global_variables_initializer())


        #Starting the training loop    
        batch_epoch = len(data_pool) // (batch_size * n_critic)
        max_it = epoch * batch_epoch
        for it in range(sess.run(it_cnt), max_it):
            sess.run(update_cnt)

            # which epoch
            epoch = it // batch_epoch
            it_epoch = it % batch_epoch + 1

            # train D
            for i in range(n_critic):  #Fist we train the discriminator for few iterations (Here I used only 1)
                # batch data
                real_ipt = data_pool.batch('img') #Read data batch
                z_ipt = np.random.normal(size=[batch_size, z_dim]) #Sample nosice input 

            
                d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z: z_ipt}) #Discriminator Gradient Update
                beta_summary_opt = sess.run(beta_summary)
                #_ = sess.run([d_step], feed_dict={real: real_ipt, z: z_ipt})
                sess.run([assign_op],feed_dict={real: real_ipt, z: z_ipt}) #Adpatively update the beta parameter

            summary_writer.add_summary(d_summary_opt, it)
            summary_writer.add_summary(beta_summary_opt, it)

            # train the geenrator (Here we have a simple generator as in normal Wgan)
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt})
            #_ = sess.run([g_step], feed_dict={z: z_ipt})
            summary_writer.add_summary(g_summary_opt, it)

            # display training progress
            if it % 100 == 0:
                print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))
             

            # saving the checpoints after every 1000 interation
            if (it + 1) % 1000 == 0:
                save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
                print('Model saved in file: % s' % save_path)

            #This is to save the  image generation during the trainign as tiles
            if (it + 1) % 100 == 0:
                z_input_sample = np.random.normal(size=[100, z_dim]) #Noise samples 
                f_sample=generator(z)
                f_sample_opt = sess.run(f_sample, feed_dict={z: z_input_sample})

                save_dir = './sample_images_while_training/mnist_wgan_gp'
                utils.mkdir(save_dir + '/')
                utils.imwrite(utils.immerge(f_sample_opt, 10, 10), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))



if __name__ == '__main__':

    #General Parameters 
    n_critic = 1 #How many iteraction we update the critic (In VDB network we update both D and G one time) 
    gpu_id = 0 #When you use multiple GPUs
    epoch = 200  
    batch_size = 16
    lr = 0.0001
    z_dim = 100 #Dimentiones of the random noise vactor

    #Sampliong Mnist Data
    utils.mkdir('./data/mnist/')
    data.mnist_download('./data/mnist')
    imgs, _, _ = data.mnist_load('./data/mnist')
    imgs.shape = imgs.shape + (1,)
    data_pool = utils.MemoryData({'img': imgs}, batch_size)


    #Variational Information Bottleneck and Training Related Paramters
    bottle_dim=512 #dimentiones of the bottleneck layer
    I_c= 0.1 #This is the information contraint (Eqation(2))
    Alpha = 1e-6 #This controls the Beta update in dual grdients
    


  

    # invoke the main function of the script
    main(epoch=epoch,batch_size=batch_size,lr=lr,z_dim=z_dim,bottle_dim=bottle_dim,i_c=I_c\
        ,alpha=Alpha,n_critic=n_critic,gpu_id=gpu_id,data_pool=data_pool)
