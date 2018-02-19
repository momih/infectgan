import numpy as np
import os
from scipy.misc import imread
from tqdm import tqdm
import random
from keras.utils import to_categorical

def get_xrays(phase='train', normal=False, test=False, folder='./data/normal/', label=0.):
    files_list = os.listdir(folder + phase + '/')
    if normal and phase == 'train':
        random.shuffle(files_list)
        files_list= files_list[:6000]
        
    images = []
    for i in tqdm(files_list):
        xray = imread(folder + phase + '/' + i, mode='L')
        images.append(xray)
    images = np.array(images)
    images = images[:,:,:, None]
    label = np.full((len(images)), label)
    
    return images, label    

def create_dataset(phase='train'):
    if phase=='train':
        trX_normal, trY_normal = get_xrays('train', normal=True)
        trX_abn, trY_abn = get_xrays('train', normal=False, folder= './data/abnormal/', label=1.)
        
        trainX = np.vstack((trX_normal, trX_abn))
        trainY = np.concatenate((trY_normal, trY_abn))
        return trainX, trainY
    
    elif phase == 'val':
        valX_normal, valY_normal = get_xrays('val')
        valX_abn, valY_abn = get_xrays('val', normal=False, folder= './data/abnormal/', label=1.)
        valX = np.vstack((valX_normal, valX_abn))
        valY = np.concatenate((valY_normal, valY_abn))
        return valX, valY
    
    elif phase == 'test':
        testX_normal, testY_normal = get_xrays('test')
        testX_abn, testY_abn = get_xrays('test', normal=False, folder= './data/abnormal/', label=1.)
        testX = np.vstack((testX_normal, testX_abn))
        testY = np.concatenate((testY_normal, testY_abn))
        return testX, testY

X_train, Y_train = create_dataset('train')
X_valid, Y_valid = create_dataset('val')

Y_train = to_categorical(Y_train)
Y_valid= to_categorical(Y_valid)

X_test, Y_test = create_dataset('test')
Y_test = to_categorical(Y_test)
np.savez('p2data.npz', X_train=X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)

np.savez('test_1.npz', X_test=X_test, Y_test=Y_test)
