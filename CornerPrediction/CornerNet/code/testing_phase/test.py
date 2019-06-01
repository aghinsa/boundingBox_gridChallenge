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
    checkpoints_dir='../../checkpoints'
   

    ## data

    iterator=dataset.make_initializable_iterator()
    sess.run(iterator.initializer)
    temp=iterator.get_next()

    images=temp[0]
 

        ##setting shape
    images=tf.reshape(images,[batch_size,img_size[0],img_size[1],3])
   
        ##setting type and normalizing
    images=tf.divide(tf.cast(images,tf.float32),255)

    


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

        master_preds=[]
        ptop=[]
        pbottom=[]
        while(True):
            try:
                batch+=1
                print(batch)
                ##predictions
                predictions_corners=sess.run(pred_corners)
                # pht=sess.run(pred_top)*255
                # phb=sess.run(pred_bottom)*255
                
                for idx in range(predictions_corners.shape[0]):
                   
                    master_preds.append(predictions_corners[idx])
                    # ptop.append(pht[idx])
                    # pbottom.append([phb[idx]])

            except tf.errors.OutOfRangeError:
                print()
                print("Total batches : {}".format(batch))               
                toc = time.clock()
                print("\t Time taken :{}".format((toc - tic) / 60))
                break
            
    print("Testing Completed")
    return master_preds,ptop,pbottom



    





img_size=(256,256)
batch_size=21
num_epochs=1000


# test_filenames=data.file_to_list('train_list.txt')
# dataset=data.create_dataset(test_filenames,batch_size=batch_size,drop=True)

test_filenames=data.file_to_list('test_list.txt')
dataset=data.create_dataset(test_filenames,batch_size=batch_size,drop=False)

predictions_list,top_list,bottom_list=test(dataset,img_size,batch_size=batch_size,filenames=test_filenames) 
predictions=np.array(predictions_list)

print('Saving')
np.save('predicted_points_to_refine_test.npy',predictions)
# np.save('predicted_points_to_refine_train.npy',predictions)
print('Saved')
    