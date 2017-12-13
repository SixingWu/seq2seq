# coding=utf-8
# http://blog.csdn.net/u012223913/article/details/75051516?locationNum=1&fps=1
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
import random

sess = tf.InteractiveSession()

mb_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def weight_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))


# Generator net
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
Yo = tf.placeholder(tf.float32, shape=[None, 10], name='Yo')
Y = Yo
W_gz1 = weight_var([10, 100, 128], 'W_gz1')
b_gz1 = bias_var([10, 128], 'b_gz1')

W_gzy3 = weight_var([128, 784], 'W_gzy3')
b_gzy3 = bias_var([784], 'b_gzy3')

theta_G = [W_gz1, b_gz1, W_gzy3, b_gzy3]


def generator(z, y):
    # Z: 16*1*100
    z_p = tf.reshape(z,[-1,1,100])
    y_ids = tf.argmax(y, axis=-1)
    # 16*100*128
    g_w = tf.reshape(tf.nn.embedding_lookup(W_gz1, y_ids),[-1, 128])
    # 16*128
    g_b = tf.reshape(tf.nn.embedding_lookup(b_gz1, y_ids),[-1,128])
    layer_z_1 = tf.nn.relu(tf.reshape(tf.matmul(z_p,g_w),[-1, 128])+ g_b)
    logits = tf.matmul(layer_z_1, W_gzy3) + b_gzy3
    G_prob = tf.nn.sigmoid(logits)

    return G_prob




def generator_old(z, y):
    layer_z_1 = tf.nn.relu(tf.matmul(z, W_gz1) + b_gz1)
    layer_y_1 = tf.nn.sigmoid(tf.matmul(y, W_gy1) + b_gy1)
    layer_zy_2 = tf.concat([layer_z_1, layer_y_1], axis=-1)
    logits = tf.matmul(layer_zy_2, W_gzy3) + b_gzy3
    G_prob = tf.nn.sigmoid(logits)

    return G_prob


# discriminater net

X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

W_dx1 = weight_var([784, 128], 'W_dx1')
b_dx1 = bias_var([128], 'b_dx1')

W_dy1 = weight_var([10, 16], 'W_dy1')
b_dy1 = bias_var([16], 'b_dy1')

W_dxy3 = weight_var([128 + 16, 1], 'W_dxy3')
b_dxy3 = bias_var([1], 'b_dxy3')

theta_D = [W_dx1, b_dx1, W_dy1, b_dy1, W_dxy3, b_dxy3]


def discriminator(x, y):
    layer_x_1 = tf.nn.relu(tf.matmul(x, W_dx1) + b_dx1)
    layer_y_1 = tf.nn.relu(tf.matmul(y, W_dy1) + b_dy1)
    layer_xy_2 = tf.concat([layer_x_1, layer_y_1], axis=-1)
    logits = tf.matmul(layer_xy_2, W_dxy3) + b_dxy3
    probs = tf.nn.sigmoid(logits)
    return logits, probs


G_sample = generator(Z, Y)
D_real, D_logit_real = discriminator(X, Y)
D_fake, D_logit_fake = discriminator(G_sample, Y)

# discriminator输出为1表示ground truth
# discriminator输出为0表示非ground truth
# 对于生成网络希望两点：
# (2)希望D_real尽可能大，这样保证正确识别真正的样本
# (1)希望D_fake尽可能小，这样可以剔除假的生成样本
D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))

# 对于判别网络, 希望D_fake尽可能大，这样可以迷惑生成网络，
G_loss = -tf.reduce_mean(tf.log(D_fake))

D_optimizer = tf.train.AdamOptimizer(0.00001).minimize(D_loss, var_list=theta_D)
G_optimizer = tf.train.AdamOptimizer(0.00001).minimize(G_loss, var_list=theta_G)

init = tf.initialize_all_variables()
saver = tf.train.Saver()
# 启动默认图
sess = tf.Session()
# 初始化
sess.run(init)


def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])


def sample_Y(m):
    '''Uniform prior for G(Z)'''
    tmp = np.zeros([m, 10])
    for i in range(m):
        index = random.randint(0, 9)
        tmp[i][index] = 1.0
    return tmp


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={
            Z: sample_Z(16, Z_dim), Yo: sample_Y(16)})  # 16*784
        try:
            fig = plot(samples)
            # plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.show()
            plt.close(fig)
        except Exception as e:
            print(e)
        finally:
            try:
                plt.close(fig)
            except:
                pass

    X_mb, Y_mb = mnist.train.next_batch(mb_size)  # ground truth

    _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={
        X: X_mb, Z: sample_Z(mb_size, Z_dim), Y: Y_mb})
    _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={
        Z: sample_Z(mb_size, Z_dim), Y: Y_mb})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
