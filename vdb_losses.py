
import tensorflow as tf
import pdb
def gradient_penalty(real, fake, Discriminator):

    #This is the usual gradient panelty fullfil the lipshitz contrain the Wgan loss.
 
    shape = tf.concat((tf.shape(real)[0:1], tf.tile([1], [real.shape.ndims - 1])), axis=0)
    alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)

    Noisy_data = real + alpha * (fake - real)
    Noisy_data.set_shape(real.get_shape().as_list())
 

    x = Noisy_data

    pred = Discriminator(x)
    gradients = tf.gradients(pred, x)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))#range(1, x.shape.ndims)))
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp


def wgan_loss(real_score, fake_score):

    #The wasserstein_distance loss 
    wasserstein_distance=tf.reduce_mean(real_score) - tf.reduce_mean(fake_score)
    return wasserstein_distance

def _bottleneck_loss(real_mus, fake_mus,real_sigmas,fake_sigmas,i_c):

    alpha=1e-8 #This is to get stable logarithm in the KL divergence equation


    #We need to input similar number of fake and real to cacluate the bottleneck loss
    mus=tf.concat(values=[real_mus,fake_mus],axis=0)
    sigmas=tf.concat(values=[real_sigmas,fake_sigmas],axis=0)

    #This equation can be obtained by simple simiplicaiton of 
    #the KL divergence equation (Similart to the variational inferece)
    kl_Divergence= 0.5* ((tf.square(mus) + tf.square(sigmas) - tf.log(tf.square(sigmas) + alpha) - 1 ))


    bottleneck_loss=tf.reduce_mean(kl_Divergence-i_c)
    #Here the i_c act as an upperbound to the kl_divergence.
    #When we later update the beta with dual gradient settings this value plays a big roll. Not like in VAE here the KL has an upper 
    #bound to work that ecourges the discriminator to learn usefull representations. 


    return bottleneck_loss
