import tensorflow as tf 
import numpy as np
import model
import utils
import data
import time
import os

def train(dataset,img_size,batch_size,num_epochs):
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
    

    ##loss 
    # tloss=utils.focal_loss(pred_top,true_heatmap_top)
    # bloss=utils.focal_loss(pred_bottom,true_heatmap_bottom)
    tloss=tf.nn.l2_loss(pred_top-true_heatmap_top)
    bloss=tf.nn.l2_loss(pred_bottom-true_heatmap_bottom)
    point_loss=tf.nn.l2_loss(pred_corners-true_points)
    
    height_loss=tf.nn.l2_loss((pred_corners[:,3]-pred_corners[:,1])-(true_points[:,3]-true_points[:,1]))
    width_loss=tf.nn.l2_loss((pred_corners[:,2]-pred_corners[:,0])-(true_points[:,2]-true_points[:,0]))



    lambda_hloss= 0
    lambda_ploss= 3
    # loss=lambda_hloss*(tloss+bloss) + lambda_ploss * (point_loss + height_loss + width_loss) 
    loss=lambda_hloss*(tloss+bloss) + lambda_ploss * (point_loss) 
    
    loss=(loss/batch_size)
    loss=tf.identity(loss)
    accuracy=utils.tf_box_mean_iou(pred_corners,true_points)
    accuracy=tf.identity(accuracy)
    learning_rate=0.00025
    # learning_rate=0.001
    optimizer= tf.train.AdamOptimizer(learning_rate).minimize(loss)

    
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(loss))
    # print(sess.run(accuracy))
    # return 

    ## summary and training
    with tf.name_scope("summaries"):
        
        total_loss_summary=tf.summary.scalar('Loss',loss)
        accuracy_summary=tf.summary.scalar('Accuracy',accuracy)
        top_loss_summary=tf.summary.scalar('TopCorner loss',tloss/batch_size)
        bottom_loss_summary=tf.summary.scalar('BottomCorner loss',bloss/batch_size)
        point_loss_summary=tf.summary.scalar('Corner loss',point_loss/batch_size)

        perfomance_summary=tf.summary.merge([total_loss_summary,accuracy_summary,
                                            top_loss_summary,bottom_loss_summary,point_loss_summary])


        # image summaries
        true_heatmap_combined=tf.add(true_heatmap_top,true_heatmap_bottom)
        pred_heatmap_combined=tf.add(pred_top,pred_bottom)
        
        is1=tf.summary.image("Input",true_heatmap_combined, max_outputs=4)
        is2=tf.summary.image("Predictions",pred_heatmap_combined, max_outputs=4)

        image_summary=tf.summary.merge([is1,is2])

    # Checkpoints

    print('Starting Training...')

    ckpt = tf.train.get_checkpoint_state(checkpoints_dir)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    is_training=True
    if(is_training):
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=2)
            # saver.restore(sess, ckpt.model_checkpoint_path)
            saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir)) # search for checkpoint file
            global_step=0
        else:
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=5, 
                                max_to_keep=2)
            global_step = 0
            
    

        sess.run(iterator.initializer)


        tf.summary.FileWriterCache.clear()
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)

        
        for epoch in range(num_epochs):
            tic = time.clock()
            print("Starting epoch {}".format(epoch + 1))
            sess.run(iterator.initializer)

            batch=1
            loss_list=[]
            accuracy_list=[]
            while(True):
                try:
                    global_step=global_step+1
    
                 
                    opt = sess.run(optimizer)
                    l=sess.run(loss)
                    acc=sess.run(accuracy)
                    loss_list.append(l)
                    accuracy_list.append(acc)
                    print("loss : {} , accuracy :{}".format(l,acc))
                    # print("loss : {} ".format(l))

                    
                    
                    #writing image (eval)summaries
                    batch=batch+1
                    
                    p_summary=sess.run(perfomance_summary)
                    train_writer.add_summary(p_summary, global_step)
                    
                    #sanity check
                    if(batch%100==0):
                        ppp=sess.run(pred_corners)
                        ppp=np.floor(ppp)
                        ppt=sess.run(true_points)
                        print("\t accuracy = {} ".format(np.mean(np.array(accuracy_list))))                    
                        print("Truth,pred at batch {}".format(batch))
                        print(ppt[0])
                        print(ppp[0])
                        print(ppt[1])
                        print(ppp[1])


                    if((batch)%100==0):
                        i_summary=sess.run(image_summary)
                        train_writer.add_summary(i_summary, global_step)

                    if((batch)%450==0):
                        saver.save(sess, os.path.join(checkpoints_dir,'model.ckpt'), 
                                    global_step=global_step)
                    
                    
                except tf.errors.OutOfRangeError:
                    saver.save(sess, os.path.join(checkpoints_dir,'model.ckpt'), 
                                global_step=global_step)
                    print()
                    print("Total batches : {}".format(batch))
                    print("\t Epoch {} summary".format(epoch + 1))
                    print("\t loss = {} ".format(np.mean(np.array(loss_list))))
                    print("\t accuracy = {} ".format(np.mean(np.array(accuracy_list))))                    
                    toc = time.clock()
                    print("\t Time taken :{}".format((toc - tic) / 60))
                    break
    print("Training Completed")
    return

    





img_size=(256,256)
batch_size=6
num_epochs=1000

train_filenames=data.file_to_list('train_list.txt')
dataset=data.create_dataset(train_filenames,batch_size=batch_size)
train(dataset,img_size,batch_size=batch_size,num_epochs=num_epochs) 
    