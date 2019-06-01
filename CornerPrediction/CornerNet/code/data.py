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
    parent='/media/aghinsa/Windows/Users/AghinShah/Documents/grid_challenge/flipGrid/dataset/images'
    
    filename=os.path.join(parent,filename.decode())
    image=cv2.imread(filename)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,(256,256))
 
    return image

def _setterY(filename):
    # returns unnormalized 

    filename=filename.decode()
    top_path="../labels/top"
    bottom_path="../labels/bottom"
    with open('labels_xy.pickle', 'rb') as handle:
        label_dict = pickle.load(handle)
    
    truth_box=label_dict[filename]
    truth_box=truth_box.reshape(-1).astype(np.float32)

    
    top_heatmap=cv2.imread(os.path.join(top_path,filename),0)
    top_heatmap=np.expand_dims(top_heatmap,axis=-1)
    bottom_heatmap=cv2.imread(os.path.join(bottom_path,filename),0)
    bottom_heatmap=np.expand_dims(bottom_heatmap,axis=-1)
    


    return (truth_box,top_heatmap,bottom_heatmap)

def create_dataset(filenames_list,batch_size,prefetch_buffer_size=8,shuffle=True):
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
    # dataset1=dataset1.map(_setterX)
    
    dataset2=tf.data.Dataset.from_tensor_slices(filenames)
    dataset2 = dataset2.map(
    lambda filename: tf.py_func(
        _setterY, [filename], [np.float32,tf.uint8,tf.uint8]))
    dataset3=tf.data.Dataset.zip((dataset1, dataset2))
    dataset3 = dataset3.batch(batch_size,drop_remainder=True)
    if shuffle:
        dataset3 = dataset3.prefetch(prefetch_buffer_size)
        dataset3 = dataset3.shuffle(buffer_size=20)
    return dataset3

if __name__ == "__main__":
    train_filenames=file_to_list('train_list.txt')
    dataset=create_dataset(train_filenames,batch_size=2)

    iterator=dataset.make_initializable_iterator()
    sess=tf.Session()
    sess.run(iterator.initializer)
    temp=iterator.get_next()

    inputs=temp[0][0]
    true_points=temp[1][0]
    true_heatmap_top=temp[1][1]
    true_heatmap_bottom=temp[1][2]

    

    print(sess.run(inputs).shape)
    print(sess.run(true_points).shape)
    print(sess.run(true_heatmap_top).shape)
    print(sess.run(true_heatmap_bottom).shape)
    print('me')
    