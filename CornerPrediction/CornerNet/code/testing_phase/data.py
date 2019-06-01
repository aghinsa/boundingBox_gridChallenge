import os 
import glob
import random
import pickle

import numpy as np
import cv2
import tensorflow as tf 
import matplotlib.pyplot as plt
import pandas as pd
import utils



def file_to_list(filenames):
    """
    return ndarray
    """
    f = open(filenames, 'r')
    text_list = []
    for line in f:
        text_list.append(line[0:-1])
    f.close()
    print("No: of files : {}".format(len(text_list)))
    return np.asarray(text_list)


def _setterX(filename):
    # parent='/media/aghinsa/DATA/workspace/flipGrid/dataset/images'
    parent='/media/aghinsa/Windows/Users/AghinShah/Documents/grid_challenge/flipGrid/dataset/images'

    filename=os.path.join(parent,filename.decode())
    image=cv2.imread(filename)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,(256,256))
 
    return image


def create_dataset(filenames_list,batch_size,prefetch_buffer_size=8,drop=True):
    """
    return dataset
    next=iterator.get_next()
    type(next) : tuple
    a[0][0] : data
    a[1][0] : label
    """
    filenames=filenames_list
    dataset1 = tf.data.Dataset.from_tensor_slices(filenames)
    dataset1 = dataset1.map(
        lambda filename: tf.py_func(
            _setterX, [filename], [tf.uint8]))
   
    dataset1 = dataset1.batch(batch_size,drop_remainder=drop)

    return dataset1

if __name__ == "__main__":
    train_filenames=file_to_list('test_list.txt')
    dataset=create_dataset(train_filenames,batch_size=2)

    iterator=dataset.make_initializable_iterator()
    sess=tf.Session()
    sess.run(iterator.initializer)
    temp=iterator.get_next()

    inputs=temp[0]
  

    

    print(sess.run(inputs).shape)
    print('me')
    