import pip

def install(package):
    pip.main(['install', package])
install('imageio')

import tensorflow as tf
import argparse
import os
from GAN_masked import InfectGAN
from shutil import copy

parser = argparse.ArgumentParser(description='')
parser.add_argument('--job-dir', dest='jobsdir', default='gs://xray8-infect', help='GCS location to write checkpoints and export models')
parser.add_argument('--dataset_name', dest='dataset_name', default='xrays', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=1, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
                    
parser.add_argument('--gf_dim', dest='gf_dim', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--infect_filters', dest='infect_filters', type=int, default=16, help='# of gen filters in first conv layer of infector')                
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
                    
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--opt', dest='opt', default='adam', help='adam or rms')
parser.add_argument('--phase', dest='phase', default='test', help='train, test')

parser.add_argument('--save_freq', dest='save_freq', type=int, default=220,
                    help='save a model every save_epoch_freq steps (does not overwrite previously saved models)')

parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=220,
                    help='save the latest model every latest_freq steps (overwrites the previous latest model)')

parser.add_argument('--print_freq', dest='print_freq', type=int, default=220,
                    help='print the debug information every print_freq iterations')

parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False,
                    help='if continue training, load the latest model: 1: true, 0: false')


parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--log_dir', dest='log', default='./logs', help='tensorboard logs saved here')

parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--gan_type', dest='gan_type', default='masked', help='which model to use: masked or extracted')

args = parser.parse_args()


if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
if not os.path.exists(args.sample_dir):
    os.makedirs(args.sample_dir)
if not os.path.exists(args.test_dir):
    os.makedirs(args.test_dir)
if not os.path.exists(args.log):
    os.makedirs(args.log)

with tf.Session() as sess:
    model = InfectGAN(sess, batch_size=args.batch_size,
                      dataset_name=args.dataset_name,
                      checkpoint_dir=args.checkpoint_dir, opt=args.opt, 
                      gf_dim=args.gf_dim, infect_filters=args.infect_filters)

if args.phase == 'train':
    model.train(args, sample_step=args.sample_freq, save_step=args.save_freq)
else:
    model.test(args)

