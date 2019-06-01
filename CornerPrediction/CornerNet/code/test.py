import tensorflow as tf 
import numpy as np
import model
import utils
import data
import time
import cv2
import os

def test(dataset,img_size,batch_size,filenames):
    """

    """
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True
    sess = tf.Session(config=sess_config)
    checkpoints_dir='../checkpoints'
    log_dir='../logs'

    ## data

    iterator=dataset.make_initializable_iterator()
    sess.run(iterator.initializer)
    temp=iterator.get_next()

    images=temp[0][0]
    true_points=temp[1][0]
    true_heatmap_top=temp[1][1]
    true_heatmap_bottom=temp[1][2]

        ##setting shape
    images=tf.reshape(images,[batch_size,img_size[0],img_size[1],3])
    true_points=tf.reshape(true_points,[batch_size,4])
    true_heatmap_top=tf.reshape(true_heatmap_top,[batch_size,img_size[0],img_size[1],1])
    true_heatmap_bottom=tf.reshape(true_heatmap_bottom,[batch_size,img_size[0],img_size[1],1])
        ##setting type and normalizing
    images=tf.divide(tf.cast(images,tf.float32),255)
    true_points=tf.cast(true_points,tf.float32)
    true_heatmap_top=tf.divide(tf.cast(true_heatmap_top,tf.float32),255)
    true_heatmap_bottom=tf.divide(tf.cast(true_heatmap_bottom,tf.float32),255)
    


    # getting feature vector
    features=model.uNet(images=images,img_size=img_size)
    features=tf.identity(features)

    #getting heatmaps
    pred_top=model.heatmap(features,mode='topLeft')
    pred_bottom=model.heatmap(features,mode="bottomRight")
    pred_top=tf.identity(pred_top)    # [?,512,512,1]
    pred_bottom=tf.identity(pred_bottom)
    pred_heatmaps=tf.concat([pred_top,pred_bottom],axis=-1)
    pred_corners=model.layer_heatmap_to_corner(pred_heatmaps)

    
   
   

    # Checkpoints

    print('Testing...')

    ckpt = tf.train.get_checkpoint_state(checkpoints_dir)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    is_testing=True

    predictions_path='../testing_phase/'

    if(is_testing):
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=2)
            # saver.restore(sess, ckpt.model_checkpoint_path)
            saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir)) # search for checkpoint file
        
        else:
            print("No checkpoints found")
            return
            
        sess.run(iterator.initializer)



        tic = time.clock()

        sess.run(iterator.initializer)

        batch=-1

        accuracy_list=[]
        master_preds=[]
        while(True):
            try:
                batch+=1

                ##predictions
                predictions_corners=sess.run(pred_corners)
                for idx in range(predictions_corners.shape[0]):
                    master_preds.append(predictions_corners[idx])   

            except tf.errors.OutOfRangeError:
                print()
                print("Total batches : {}".format(batch))               
                toc = time.clock()
                print("\t Time taken :{}".format((toc - tic) / 60))
                break
            break
    print("Testing Completed")
    return master_preds



    





img_size=(256,256)
batch_size=13
num_epochs=1000

test_filenames=data.file_to_list('test_list.txt')
dataset=data.create_dataset(test_filenames,batch_size=batch_size,shuffle=False)
predictions_list=test(dataset,img_size,batch_size=batch_size,filenames=test_filenames) 
predictions=np.array(predictions_list)
print('Saving')
print(len(predictions_list)-predictions.shape[0])
np.save('predicted_points.npy',predictions)
print('Saved')
    