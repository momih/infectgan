from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from six.moves import xrange
from tensorflow.contrib.layers import layer_norm

from ops import batch_norm, conv2d, deconv2d, linear, lrelu
from utils import save_images, read_data, save_stats, read_masked


class InfectGAN(object):
    def __init__(self, sess,
                 image_size=256,
                 batch_size=1,
                 sample_size=1,
                 output_size=256,
                 gf_dim=1,
                 df_dim=64,
                 l1_lambda=100,
                 defect_lambda=0.5,
                 wgan_lambda=10.0,
                 input_c_dim=1,
                 output_c_dim=1,
                 latent_bbox_size=256,
                 dataset_name="xray8",
                 checkpoint_dir=None,
                 loss="wgan",
                 paired=False,
                 contrast=False,
                 bbox_channels=2,
                 infect_filters=1,
                 infector_skip=False,
                 opt='adam'):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
            :type defect_lambda: object
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.checkpoint_dir = checkpoint_dir

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.bbox_channels = bbox_channels

        self.L1_lambda = l1_lambda
        self.defect_lamba = defect_lambda
        self.wgan_lambda = wgan_lambda
        
        self.latent_bbox_size = latent_bbox_size
        self.infect_filters = infect_filters
        self.infector_skip = infector_skip
        
        self.loss = loss
        self.paired = paired
        self.contrast = contrast
        self.opt = opt
        
        # batch normalization : deals with poor initialization helps gradient flow
        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.infect_bn_e2 = batch_norm(name='i_bn_e2')
        self.infect_bn_e3 = batch_norm(name='i_bn_e3')
        self.infect_bn_e4 = batch_norm(name='i_bn_e4')
        self.infect_bn_e5 = batch_norm(name='i_bn_e5')
        self.infect_bn_e6 = batch_norm(name='i_bn_e6')
        self.infect_bn_e7 = batch_norm(name='i_bn_e7')
        self.infect_bn_e8 = batch_norm(name='i_latent_bbox')

        self.infect_bn_d1 = batch_norm(name='i_bn_d1')
        self.infect_bn_d2 = batch_norm(name='i_bn_d2')
        self.infect_bn_d3 = batch_norm(name='i_bn_d3')
        self.infect_bn_d4 = batch_norm(name='i_bn_d4')
        self.infect_bn_d5 = batch_norm(name='i_bn_d5')
        self.infect_bn_d6 = batch_norm(name='i_bn_d6')
        self.infect_bn_d7 = batch_norm(name='i_bn_d7')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.normal_xrays = tf.placeholder(tf.float32,
                                           [self.batch_size, self.image_size, self.image_size, self.input_c_dim],
                                           name='normal_xrays')
        self.real_abnormal = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size, self.image_size, self.input_c_dim],
                                            name='abnormal_xrays')
        
        self.masked_bbox = tf.placeholder(tf.float32,
                                          [self.batch_size, self.latent_bbox_size,
                                          self.latent_bbox_size, self.bbox_channels],
                                          name='masked_abnormal')
        
        self.fake_abnormal, self.reconstructed_bbox = self.generator(image=self.normal_xrays, mask=self.masked_bbox)

        if self.loss == "adversarial":
            if self.paired:
                # Pair normal images with real and generated abnormal images
                self.real_abnormal_to_discrm = tf.concat([self.normal_xrays, self.real_abnormal], 3)
                self.fake_abnormal_to_discrm = tf.concat([self.normal_xrays, self.fake_abnormal], 3)
            else:
                self.real_abnormal_to_discrm = self.real_abnormal
                self.fake_abnormal_to_discrm = self.fake_abnormal

            # Get discriminator logits for real images             
            self.D, self.real_logits = self.discriminator(self.real_abnormal_to_discrm, reuse=False)
            
            # Get discriminator logits for fake images
            self.D_, self.fake_logits = self.discriminator(self.fake_abnormal_to_discrm, reuse=True)
            
            # Discriminator loss
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits,
                                                                                      labels=tf.ones_like(self.D)))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,
                                                                                      labels=tf.zeros_like(self.D_)))
            
            self.d_loss = self.d_loss_real + self.d_loss_fake

            # Do we do add contrast loss?
            if self.contrast:
                self.abnormal_contrast = tf.concat([self.real_abnormal, self.fake_abnormal], 3)
                self.normal_contrast = tf.concat([self.normal_xrays, self.fake_abnormal], 3)
               
                # Get discriminator logits for abnormal contrast             
                self.D_c, self.ac_logits = self.discriminator(self.abnormal_contrast, reuse=False)
                
                # Get discriminator logits for normal contrast
                self.D_c_, self.nc_logits = self.discriminator(self.normal_contrast, reuse=True)    
                
                d_loss_ac = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.ac_logits,
                                                                                   labels=tf.ones_like(self.D_c)))
                d_loss_nc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.nc_logits,
                                                                                   labels=tf.zeros_like(self.D_c_)))

                self.d_loss = self.d_loss + d_loss_ac + d_loss_nc
        
            # Generator loss
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits,
                                                                                 labels=tf.ones_like(self.D_))) \
                          + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_abnormal - self.fake_abnormal)) \
                          + self.defect_lamba * tf.reduce_mean(tf.abs(self.masked_bbox - self.reconstructed_bbox))
    
        elif self.loss == "wgan":
        
            # GP function
            def gradient_penalty(real, fake):
                def interpolate(a, b):
                    shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
                    alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
                    inter = a + alpha * (b - a)
                    inter.set_shape(a.get_shape().as_list())
                    return inter
        
                x = interpolate(real, fake)
                unused, pred = self.discriminator(x, reuse=True)
                gradients = tf.gradients(pred, [x])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, x.shape.ndims)))
                gp = tf.reduce_mean((slopes - 1.)**2)
                return gp          
            
            # Get discriminator logits for real images     
            self.D, self.real_logits = self.discriminator(self.real_abnormal, reuse=False)
            
            # Get discriminator logits for fake images  
            self.D_, self.fake_logits = self.discriminator(self.fake_abnormal, reuse=True)
        
            # Discriminator loss
            
            self.d_loss_real = tf.reduce_mean(self.real_logits)
            self.d_loss_fake = tf.reduce_mean(self.fake_logits)
            self.grad_p = gradient_penalty(self.real_abnormal, self.fake_abnormal)
            
            self.d_loss = self.d_loss_fake - self.d_loss_real + (self.grad_p * self.wgan_lambda)

            # Generator loss
            self.g_loss = -tf.reduce_mean(self.fake_logits) + self.defect_lamba * tf.reduce_mean(tf.abs(self.masked_bbox - self.reconstructed_bbox))

        self.generated_sample = self.generator(self.normal_xrays, self.masked_bbox, reuse=True)

        # Summary ops
        self.g_loss_summary = tf.summary.scalar("g_loss", self.g_loss)
        
        self.d_loss_real_summary = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_summary = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        
        self.d_loss_summary = tf.summary.scalar("d_loss", self.d_loss)

        self.d_summary = tf.summary.histogram("real_logits", self.D)
        self.d__summary = tf.summary.histogram("fake_logits", self.D_)
        self.fake_abnormal_summary = tf.summary.image("fake_abnormal", self.fake_abnormal)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name or 'i_' in var.name]

        self.saver = tf.train.Saver()
        self.tf_merged_summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./logs/', self.sess.graph)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
                scope.reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(layer_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(layer_norm(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(layer_norm(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, image, mask, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            
            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # =============================================================================
            #     Infector network
            # =============================================================================

            # ======================= Encoding process for infection ======================

            # masked image is (256 x 256 x 2)
            infector_e1 = conv2d(mask, self.infect_filters, name='i_e1_conv')
            # e1 is (128 x 128 x self.infect_filters)

            infector_e2 = self.infect_bn_e2(conv2d(lrelu(infector_e1, name='i_e2_lrelu'), self.infect_filters * 2, name='i_e2_conv'))
            # e2 is (64 x 64 x self.infect_filters*2)

            infector_e3 = self.infect_bn_e3(conv2d(lrelu(infector_e2), self.infect_filters * 4, name='i_e3_conv'))
            # e3 is (32 x 32 x self.infect_filters*4)

            infector_e4 = self.infect_bn_e4(conv2d(lrelu(infector_e3), self.infect_filters * 8, name='i_e4_conv'))
            # e4 is (16 x 16 x self.infect_filters*8)

            infector_e5 = self.infect_bn_e5(conv2d(lrelu(infector_e4), self.infect_filters * 8, name='i_e5_conv'))
            # e5 is (8 x 8 x self.infect_filters*8)

            infector_e6 = self.infect_bn_e6(conv2d(lrelu(infector_e5), self.infect_filters * 8, name='i_e6_conv'))
            # e6 is (4 x 4 x self.infect_filters*8)

            infector_e7 = self.infect_bn_e7(conv2d(lrelu(infector_e6), self.infect_filters * 8, name='i_e7_conv'))
            # e7 is (2 x 2 x self.infect_filters*8)

            latent_masked = self.infect_bn_e8(conv2d(lrelu(infector_e7), self.infect_filters * 8, name='i_latent_mask_conv'))
            # latent_masked is (1 x 1 x self.infect_filters*8)

            # ======================= Decoding process for infection ======================

            i_d7 = deconv2d(input_=tf.nn.relu(latent_masked),
                            output_shape=[self.batch_size, s128, s128, self.infect_filters * 8],
                            name='i_d7')
            infect_d7 = tf.nn.dropout(self.infect_bn_d7(i_d7), 0.5)
            if self.infector_skip:
                infect_d7 = tf.concat([infect_d7, infector_e7], 3)
            # d7 is (2 x 2 x self.infect_filters*8*2)

            i_d6 = deconv2d(input_=tf.nn.relu(infect_d7),
                            output_shape=[self.batch_size, s64, s64, self.infect_filters * 8],
                            name='i_d6')
            infect_d6 = tf.nn.dropout(self.infect_bn_d6(i_d6), 0.5)
            if self.infector_skip:
                infect_d6 = tf.concat([infect_d6, infector_e6], 3)
                # d6 is (4 x 4 x self.infect_filters*8*2)

            i_d5 = deconv2d(input_=tf.nn.relu(infect_d6),
                            output_shape=[self.batch_size, s32, s32, self.infect_filters * 8],
                            name='i_d5')
            infect_d5 = tf.nn.dropout(self.infect_bn_d5(i_d5), 0.5)
            if self.infector_skip:
                infect_d5 = tf.concat([infect_d5, infector_e5], 3)
                # d5 is (8 x 8 x self.infect_filters*8*2)

            i_d4 = deconv2d(input_=tf.nn.relu(infect_d5),
                            output_shape=[self.batch_size, s16, s16, self.infect_filters * 8],
                            name='i_d4')
            infect_d4 = tf.nn.dropout(self.infect_bn_d4(i_d4), 0.5)
            if self.infector_skip:
                infect_d4 = tf.concat([infect_d4, infector_e4], 3)
                # d4 is (16 x 16 x self.infect_filters*8*2)

            i_d3 = deconv2d(input_=tf.nn.relu(infect_d4),
                            output_shape=[self.batch_size, s8, s8, self.infect_filters * 4],
                            name='i_d3')
            infect_d3 = tf.nn.dropout(self.infect_bn_d3(i_d3), 0.5)
            if self.infector_skip:
                infect_d3 = tf.concat([infect_d3, infector_e3], 3)
                # d3 is (32 x 32 x self.infect_filters*8*2)

            i_d2 = deconv2d(input_=tf.nn.relu(infect_d3),
                            output_shape=[self.batch_size, s4, s4, self.infect_filters * 2],
                            name='i_d2')
            infect_d2 = tf.nn.dropout(self.infect_bn_d2(i_d2), 0.5)
            if self.infector_skip:
                infect_d2 = tf.concat([infect_d2, infector_e2], 3)
                # d2 is (64 x 64 x self.infect_filters*8*2)

            i_d1 = deconv2d(input_=tf.nn.relu(infect_d2),
                            output_shape=[self.batch_size, s2, s2, self.infect_filters * 1],
                            name='i_d1')
            infect_d1 = tf.nn.dropout(self.infect_bn_d1(i_d1), 0.5)
            if self.infector_skip:
                infect_d1 = tf.concat([infect_d1, infector_e1], 3)
                # d1 is (128 x 128 x self.infect_filters*8*2)

            mask_output = deconv2d(tf.nn.relu(infect_d1),
                                            [self.batch_size, s, s, self.bbox_channels],
                                            name='i_mask_output')
            # mask_output is (256 x 256 x self.output_c_dim)

            # =============================================================================
            #     Healthy network
            # =============================================================================

            # ======================= Encoding process for healthy ======================

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            latent_image = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            # ======================= Decoding process for healthy ======================

            self.d7 = deconv2d(tf.nn.relu(latent_image),
                               [self.batch_size, s128, s128, self.gf_dim * 8],
                               name='g_d7')
            d7 = tf.nn.dropout(self.g_bn_d7(self.d7), 0.5)
            d7 = tf.concat([d7, e7, infect_d7], 3, name='skip_7')
            # d7 is (2 x 2 x self.gf_dim*8*2)

            self.d6 = deconv2d(tf.nn.relu(d7),
                               [self.batch_size, s64, s64, self.gf_dim * 8],
                               name='g_d6')
            d6 = tf.nn.dropout(self.g_bn_d6(self.d6), 0.5)
            d6 = tf.concat([d6, e6, infect_d6], 3, name='skip_6')
            # d6 is (4 x 4 x self.gf_dim*8*2)

            self.d5 = deconv2d(tf.nn.relu(d6),
                               [self.batch_size, s32, s32, self.gf_dim * 8],
                               name='g_d5')
            d5 = tf.nn.dropout(self.g_bn_d5(self.d5), 0.5)
            d5 = tf.concat([d5, e5, infect_d5], 3, name='skip_5')
            # d5 is (8 x 8 x self.gf_dim*8*2)

            self.d4 = deconv2d(tf.nn.relu(d5),
                               [self.batch_size, s16, s16, self.gf_dim * 8],
                               name='g_d4')
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4, infect_d4], 3, name='skip_4')
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d3 = deconv2d(tf.nn.relu(d4),
                               [self.batch_size, s8, s8, self.gf_dim * 4],
                               name='g_d3')
            d3 = self.g_bn_d3(self.d3)
            d3 = tf.concat([d3, e3, infect_d3], 3, name='skip_3')
            # d3 is (32 x 32 x self.gf_dim*8*2)

            self.d2 = deconv2d(tf.nn.relu(d3),
                               [self.batch_size, s4, s4, self.gf_dim * 2],
                               name='g_d2')
            d2 = self.g_bn_d2(self.d2)
            d2 = tf.concat([d2, e2, infect_d2], 3, name='skip_2')
            # d2 is (64 x 64 x self.gf_dim*8*2)

            self.d1 = deconv2d(tf.nn.relu(d2),
                               [self.batch_size, s2, s2, self.gf_dim],
                               name='g_d1')
            d1 = self.g_bn_d1(self.d1)
            d1 = tf.concat([d1, e1, infect_d1], 3, name='skip_1')
            # d1 is (128 x 128 x self.gf_dim*8*2)

            self.generated_image = deconv2d(tf.nn.relu(d1),
                                            [self.batch_size, s, s, self.output_c_dim],
                                            name='g_gen_image')

            # generated_image is (256 x 256 x self.output_c_dim)

            if not reuse:
                return tf.nn.tanh(self.generated_image), mask_output
            else:
                return tf.nn.tanh(self.generated_image)

    def sample_model(self, sample_dir, epoch, idx, nrows=2, data_type='val/'):
        n_val = read_data('normal/')
        d_val = read_masked('masked/')

        # Get batches
        n_batch = n_val.next_batch(self.batch_size, which=data_type, labels=True)
        d_batch = d_val.next_batch(self.batch_size, which=data_type, labels=True)

        # Run generator
        samples = self.sess.run(self.generated_sample,
                                feed_dict={self.normal_xrays: n_batch[0],
                                           self.masked_bbox: d_batch[0]})

        # Save the generated images
        save_images(samples, [nrows, self.batch_size//nrows], "./{}/train_{:02d}_{:04d}.png".format(sample_dir, epoch, idx))
        print('Samples saved')
        
    def train(self, args, sample_step=20, save_step=200):
        """
        Train GAN
        Data to generator -
        * Normal Xrays - 256x256, 1 channel - 10k - read in as normal
        * Bounding box defect - 32x32, 3 channels - 10k - read in as defect

        Data to discriminator -
        * Abnormal Xrays - 256x256, 1 channel - 5k - read in as abnormal
        """
        if self.opt == 'adam':
            d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                              .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                              .minimize(self.g_loss, var_list=self.g_vars)
        elif self.opt == 'rms':           
            d_optim = tf.train.RMSPropOptimizer(args.lr) \
                              .minimize(self.d_loss, var_list=self.d_vars)
            g_optim = tf.train.RMSPropOptimizer(args.lr) \
                              .minimize(self.g_loss, var_list=self.g_vars)

        print('Init')
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        print('Summary merging')
        self.g_summary = tf.summary.merge([self.d__summary, self.fake_abnormal_summary,
                                           self.d_loss_fake_summary, self.g_loss_summary])
        self.d_summary = tf.summary.merge([self.d_summary, self.d_loss_real_summary, self.d_loss_summary])

        counter = 1
        start_time = time.time()

        print('Reading in generator data')
        # read_data returns a DataSet object with next_batch method
        normal = read_data('normal/')
        defect = read_masked('masked/')

        # reading in discriminator data
        abnormal = read_data('abnormal/')

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(args.epoch):
            batch_idxs = 11000 // self.batch_size

            for idx in xrange(0, batch_idxs):
                print("\nStarting step {:4d} of epoch {:2d}".format(idx, epoch))
              

                print('\nGetting batches')
                normal_batch_togen = normal.next_batch(self.batch_size)
                defect_batch_togen = defect.next_batch(self.batch_size)
                abnormal_batch_todisc = abnormal.next_batch(self.batch_size)

                input_feed_dict = {self.normal_xrays: normal_batch_togen,
                                   self.masked_bbox: defect_batch_togen,
                                   self.real_abnormal: abnormal_batch_todisc}
                print('\nUpdating G network')
                _, summary_str = self.sess.run([g_optim, self.g_summary],
                                               feed_dict=input_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Run critic five times to make sure that d_loss does not go to zero (different from paper)
                print('Updating D network 5 times')
                for i in range(5):
                    _, summary_str = self.sess.run([d_optim, self.d_summary],
                                                   feed_dict=input_feed_dict)
                self.writer.add_summary(summary_str, counter)

                errD = self.d_loss.eval(session=self.sess, feed_dict=input_feed_dict)

                errG = self.g_loss.eval(session=self.sess, feed_dict=input_feed_dict)

                counter += 1
                print("\nEpoch: [%2d] [%4d/%4d] time: %4.4f, Disc loss: %.8f, Gen loss: %.8f"
                    % (epoch, idx, batch_idxs,
                       time.time() - start_time, errD, errG))

                # See sample images every sample_step steps
                if np.mod(counter, sample_step) == 0:
                    print('\nSampling images from model\n')
                    self.sample_model(args.sample_dir, epoch, idx)

                # Save model after half epoch
                if np.mod(counter, save_step) == 0:
                    print('\nSaving model\n')
                    self.save(args.checkpoint_dir, counter)
        
               

    def save(self, checkpoint_dir, step):
        model_name = "GAN_masked"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print("Succesfully restored from {}".format(ckpt_name))
            return True
        else:
            return False

    def test(self, args, nrows=2, stats_file='model_stats'):
        """Test GAN"""
        print('Init')
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # read_data returns a DataSet object with next_batch method
        normal = read_data('normal/')
        defect = read_masked('masked/')

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed. Continuing")
        start = time.time()
        for i in xrange(0, 3000):
            print('Generating samples for batch %2d, time: %4.4f' % (i, time.time() - start))
            normal_test_batch = normal.next_batch(1, which='test/', labels=True)
            defect_test_batch = defect.next_batch(1, which='test/', labels=True)
            file_combinations = zip(normal_test_batch[1], defect_test_batch[1])

            self._samples = self.sess.run(self.generated_sample,
                                    feed_dict={self.normal_xrays: normal_test_batch[0],
                                               self.masked_bbox: defect_test_batch[0]})
            image_filename = './{}/test_{:04d}.png'.format(args.test_dir, i)
            save_images(images=self._samples, size=[1, 1], image_path=image_filename)
            save_stats(filename=stats_file, image_name=image_filename, labels=file_combinations)
