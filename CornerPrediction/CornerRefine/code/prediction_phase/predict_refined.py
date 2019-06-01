import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as dense
import utils
import time
import os 


def refine(inputs):
    net=dense(inputs,num_outputs=8)
    net=dense(inputs,num_outputs=32)
    net=dense(net,num_outputs=256)
    net=dense(net,num_outputs=256)
    net=dense(net,num_outputs=128)
    net=dense(net,num_outputs=64)
    net=dense(net,num_outputs=8)
    net=dense(net,num_outputs=4,activation_fn=None)
    return net

def predict(inputs):


    inputs=tf.cast(inputs,tf.float32)

    preds=refine(inputs)
    preds=tf.identity(preds)
    sess=tf.Session()
 
    
    checkpoints_dir='../../checkpoints'
    
   
    # Checkpoints

    print('Starting Training...')

    ckpt = tf.train.get_checkpoint_state(checkpoints_dir)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

   
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=2)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir)) # search for checkpoint file
        print("Restored")
        global_step=0
    else:
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=5, 
                            max_to_keep=2)
        print("Not found")
        global_step = 0
            
    
       
    predictions=sess.run(preds)
    print("Training Completed")
    return predictions

###main
# inputs=np.random.rand(5,4).astype(np.float32)
# truth=np.random.rand(5,4).astype(np.float32)
# inputs=tf.convert_to_tensor(inputs)
# truth=tf.convert_to_tensor(truth)
# num_epochs=2



inputs=np.load('predicted_points_to_refine_test.npy')
batch_size=inputs.shape[0]

inputs=tf.convert_to_tensor(inputs)
predictions=predict(inputs)
keep_prob_rate=1
np.save('predicted_points_submitions_ready.npy',predictions)
print("Complete")

