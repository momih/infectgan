from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import random_seed
from scipy.misc import imread
from tensorflow.python.lib.io import file_io
import numpy as np
import os

class Xrays(object):
    def __init__(self,
                 directory,
                 seed=None):
        """
        Construct a DataSet
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.directory = directory

    @property
    def filenames(self):
        return self._filenames

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def read_img(self, image_name, squash=(-1, 1)):

        """
        Takes an image filename and returns a preprocessed image array
        :param image_name: filename of image
        :param squash: range to be squashed to

        :return: image_arr: np array of images of dim (n, height, width, channels)
        """
        full_path = self.data_dir + image_name
        image_arr = imread(full_path, mode='L')

        # Squash image to [-1, 1]
        image_arr = image_arr.astype(np.float32) * (squash[-1] - squash[0]) / 255.0 + squash[0]

        # Add channels dimension to grayscale image
        image_arr = image_arr[:, :, None]
        return image_arr

    def return_batch_filenames(self, batch_size, which_dir='train/', shuffle=True):
        """
        Return the next `batch_size` filenames from this data set
        """
        self.data_dir = self.directory + which_dir
        self._filenames = os.listdir(self.data_dir)
        self._num_examples = len(self.filenames)

        start = self._index_in_epoch

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._filenames = [self.filenames[i] for i in perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            filenames_rest_part = self._filenames[start:self._num_examples]

            # Shuffle the data for next epoch
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._filenames = [self.filenames[i] for i in perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            filenames_new_part = self._filenames[start:end]
            return filenames_rest_part + filenames_new_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._filenames[start:end]

    def next_batch(self, batch_size, which='train/', labels=False):
        batch_filenames = self.return_batch_filenames(batch_size, which)
        images = np.stack([self.read_img(x) for x in batch_filenames])
        if labels:
            return images, batch_filenames
        else:
            return images



class Masked(object):
    def __init__(self,
                 directory,
                 mask_range=(-1,1),
                 seed=None):
        """
        Construct a DataSet
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.directory = directory
        self.mask_range = mask_range
        self.infect_dir = directory + 'infect/'

    @property
    def filenames(self):
        return self._filenames

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def read_img_and_mask(self, image_name, squash=(-1, 1)):
    
        """
        Takes an image filename and returns a preprocessed image array
        :param image_name: filename of image
        :param squash: range to be squashed to
    
        :return: image_arr: np array of images of dim (n, height, width, channels)
        """
    
        full_path = self.data_dir + image_name
        infect_path = self.infect_dir + image_name[:9] + '.png'
        
        mask = imread(full_path, mode='L')
        img = imread(infect_path, mode='L')
        
        mask = np.where(mask<240, 0, 255)
        # Squash image to [-1, 1]
        img = img.astype(np.float32) * (squash[-1] - squash[0]) / 255.0 + squash[0]
        mask = mask.astype(np.float32) * (self.mask_range[-1] - self.mask_range[0]) / 255.0 + self.mask_range[0]
    
        # Add channels dimension to grayscale image
        img = img[:, :, None]
        mask = mask[:, :, None]
    
        ret = np.concatenate((mask, img), axis=2)
        return ret
    

    def return_batch_filenames(self, batch_size, which_dir='train/', shuffle=True):
        """
        Return the next `batch_size` filenames from this data set
        """
        self.data_dir = self.directory + which_dir
        self._filenames = os.listdir(self.data_dir)
        self._num_examples = len(self.filenames)

        start = self._index_in_epoch

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._filenames = [self.filenames[i] for i in perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            filenames_rest_part = self._filenames[start:self._num_examples]

            # Shuffle the data for next epoch
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._filenames = [self.filenames[i] for i in perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            filenames_new_part = self._filenames[start:end]
            return filenames_rest_part + filenames_new_part
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._filenames[start:end]

    def next_batch(self, batch_size, which='train/', labels=False):
        batch_filenames = self.return_batch_filenames(batch_size, which)
        images = np.stack([self.read_img_and_mask(x) for x in batch_filenames])
        if labels:
            return images, batch_filenames
        else:
            return images