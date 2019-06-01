import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as dense
import utils
import time
import os 

def cust_loss(preds,truth,weight,delta):
    mask=tf.greater(tf.abs(truth-preds),delta)
    p=tf.boolean_mask(preds,mask=mask)
    t=tf.boolean_mask(truth,mask=mask)    
    p=tf.multiply(tf.abs(t-p),weight)
    p=tf.reduce_mean(p)
    return p
def refine(inputs,keep_prob_rate):
    net=dense(inputs,num_outputs=8)
    net=dense(inputs,num_outputs=32)
    net=dense(net,num_outputs=256)
    net=dense(net,num_outputs=256)
    net=dense(net,num_outputs=128)
    net=dense(net,num_outputs=64)
    net=dense(net,num_outputs=8)
    net=dense(net,num_outputs=4,activation_fn=None)

    return net

def train(inputs,truth,num_epochs,batch_size,keep_prob_rate):


    inputs=tf.cast(inputs,tf.float32)
    truth=tf.cast(truth,tf.float32)
    preds=refine(inputs,keep_prob_rate)
    preds=tf.identity(preds)
    sess=tf.Session()




    true_height=truth[:,3]-truth[:,1]
    true_width=truth[:,2]-truth[:,0]
    preds_height= preds[:,3]-preds[:,1]
    preds_width=preds[:,2]-preds[:,0]

    # pl=tf.nn.l2_loss(preds-truth)
    # pl=tf.sqrt(pl)
    # hl=tf.nn.l2_loss(true_height-preds_height)
    # hl=tf.sqrt(hl)
    # wl=tf.nn.l2_loss(true_width-preds_width)
    # wl=tf.sqrt(wl)

    # wl=wl/640*640*640

    
    # pl=tf.losses.huber_loss(predictions=preds,labels=truth,delta=8.0,weights=5)
    # hl=tf.losses.huber_loss(true_height,preds_height,delta=5.0)
    # wl=tf.losses.huber_loss(true_width,preds_width,delta=5.0)
    pl=cust_loss(preds,truth,weight=10,delta=4)
    loss=pl
    # loss=hl+wl
    # loss=pl + hl + wl
    # loss= loss/batch_size
   
    loss=loss
    loss=tf.identity(loss)

    accuracy=utils.tf_box_mean_iou(preds,truth)
    accuracy=tf.identity(accuracy)
    # learning_rate=0.00025
    learning_rate=0.0001
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    checkpoints_dir='../checkpoints'
    log_dir='../logs'
    with tf.name_scope("summaries"):
        
        total_loss_summary=tf.summary.scalar('Loss',loss)
        accuracy_summary=tf.summary.scalar('Accuracy',accuracy)
     
        perfomance_summary=tf.summary.merge([total_loss_summary,accuracy_summary])
    # Checkpoints

    print('Starting Training...')

    ckpt = tf.train.get_checkpoint_state(checkpoints_dir)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    is_training=True
    if(is_training):
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=2)
            saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir)) # search for checkpoint file
            print("Restored")
            global_step=0
        else:
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=5, 
                                max_to_keep=2)
            global_step = 0
            
    
        tf.summary.FileWriterCache.clear()
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)

        tic = time.clock()
        
        for epoch in range(1,num_epochs):
            if (epoch%100==0):
                print(" epoch {}".format(epoch))
                print("Truth,pred")
                ppp=sess.run(preds)
                ppt=sess.run(truth)
                print(ppt[0])
                print(ppp[0])
                print(ppt[1])
                print(ppp[1])
            while(True):
                
                global_step=global_step+1

                
                opt = sess.run(optimizer)
                l=sess.run(loss)
                acc=sess.run(accuracy)
            
                
                # if (epoch%50)==0:
                
                print("loss : {} , accuracy : {} ".format(l,acc))
                p_summary=sess.run(perfomance_summary)
                train_writer.add_summary(p_summary, global_step)
                    
                    
                if (epoch%100==0):
                    saver.save(sess, os.path.join(checkpoints_dir,'model.ckpt'), 
                                global_step=global_step)
                                
                    toc = time.clock()
                    print("\t Time taken for 1000 epochs :{}".format((toc - tic) / 60))
                    tic=time.clock()
                break
    print("Training Completed")
    return

###main
# inputs=np.random.rand(5,4).astype(np.float32)
# truth=np.random.rand(5,4).astype(np.float32)
# inputs=tf.convert_to_tensor(inputs)
# truth=tf.convert_to_tensor(truth)
# num_epochs=2



inputs=np.load('predicted_points_to_refine_train.npy')
batch_size=inputs.shape[0]
truth=np.load('true_points.npy')



# print(truth.shape)
# print(inputs.shape)

inputs=tf.convert_to_tensor(inputs)
truth=tf.convert_to_tensor(truth)
num_epochs=50000
keep_prob_rate=1
train(inputs,truth,num_epochs,batch_size,keep_prob_rate)

