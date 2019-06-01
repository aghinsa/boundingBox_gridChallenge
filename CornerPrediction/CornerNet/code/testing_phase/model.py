import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.framework as framework
import utils



def uNet(images,img_size=(512,512),
        out_channels=128,
        views=2,
        normalizer_fn=tf_layers.batch_norm,
        activation=tf.nn.leaky_relu):
    """
    images:n*h*x*c

    Returns:
    [?,h,w,64]
    list of heatmaps:
        heatmap[0]=top
        heatmap[1]=bottom
    """
    with tf.name_scope("model"):
        images=tf.reshape(images,[-1,img_size[0],img_size[1],3])
        #images = tf.cast(images, tf.float32)
        with tf.variable_scope("encoder"):
            with framework.arg_scope([tf_layers.conv2d],
                                    kernel_size=3, stride=2, normalizer_fn=normalizer_fn,
                                    activation_fn=tf.nn.leaky_relu, padding="same"):
                e1 = tf_layers.conv2d(images, num_outputs=64)  # 256 x 256 x 64
                
                e2 = tf_layers.conv2d(e1, num_outputs=128)  # 128x128x128
              
                e3 = tf_layers.conv2d(e2, num_outputs=128)  # 64x64x256  
                      
                e4 = tf_layers.conv2d(e3, num_outputs=256)  # 32x32x512    
                     
                e5 = tf_layers.conv2d(e4, num_outputs=512)  # 16x16x512    
                     
                e6 = tf_layers.conv2d(e5, num_outputs=512)  # 8X8X512        
                
                e7 = tf_layers.conv2d(e6, num_outputs=512)  # 4X4X512
              
                encoded = tf_layers.conv2d(e7, num_outputs=512)  # 2X2X512
            
    
        
        with tf.name_scope("decoders"):
                d6 = utils.upsample(encoded, 512)  # 4X4x512
                d5 = utils.upsample(tf.concat([d6, e7], 3), 512) # 8X8X512
                d4 = utils.upsample(tf.concat([d5, e6], 3), 256)  # 16x16x512
                d3 = utils.upsample(tf.concat([d4, e5], 3), 256)  # 32x32x256
                d2 = utils.upsample(tf.concat([d3, e4], 3), 128)  # 64x64x128
                d1 = utils.upsample(tf.concat([d2, e3], 3), 128)  # 128x128x64
                d0 = utils.upsample(tf.concat([d1, e2], 3), 128)  # 256x256x64
            
        
                decoded = utils.upsample(
                    tf.concat([d0,e1],3),out_channels,
                    activation_fn=tf.nn.relu,normalizer_fn=tf_layers.batch_norm)  # 512x512xout_channels

        return decoded


def heatmap(inputs,mode):
    """
    inputs: output of uNet
    mode[str]:'topLeft','bottomRight'
    return [?,h,w,1] sigmoided
    """
    normalizer_fn=tf_layers.batch_norm
    activation=tf.nn.leaky_relu
    with framework.arg_scope([tf_layers.conv2d],
                                    kernel_size=3, stride=1, normalizer_fn=normalizer_fn,
                                    activation_fn=tf.nn.leaky_relu, padding="same"):
    
        net=utils.corner_pool(inputs,mode=mode)
        net=tf_layers.conv2d(net, num_outputs=128)
        net=tf.add(net,inputs)
        net=tf_layers.conv2d(net, num_outputs=64)
        net=tf_layers.conv2d(net, num_outputs=32)
        net=tf_layers.conv2d(net, num_outputs=1,activation_fn=tf.sigmoid)
    return net

def layer_heatmap_to_corner(heatmap):
    """
    [?,h,w,2]
    """
    net=tf_layers.conv2d(heatmap,kernel_size=5,num_outputs=64,stride=4,activation_fn=tf.nn.relu)
    net=tf_layers.conv2d(net,kernel_size=5,num_outputs=128,stride=4,activation_fn=tf.nn.relu)
    net=tf_layers.conv2d(net,kernel_size=5,num_outputs=256,stride=4,activation_fn=tf.nn.relu)
    net=tf_layers.conv2d(net,kernel_size=5,num_outputs=256,stride=4,activation_fn=tf.nn.relu)
    net=tf.squeeze(net,axis=[1,2])
    net=tf.contrib.layers.fully_connected(net,64,activation_fn=tf.nn.relu)
    net=tf.contrib.layers.fully_connected(net,4,activation_fn=tf.nn.relu)    
    return net



if __name__=="__main__":
    batch_size=5
    img_size=(256,256)
    truth_top=np.random.rand(batch_size,img_size[0],img_size[0],1).astype(np.float32)
    truth_bottom=np.random.rand(batch_size,img_size[0],img_size[0],1).astype(np.float32)
    truth_points=np.random.rand(batch_size,4).astype(np.float32)

    images= np.random.rand(batch_size,img_size[0],img_size[0],3).astype(np.float32)

    features=uNet(images=images,img_size=(img_size[0],img_size[0]))

    top_heatmap=heatmap(features,mode='topLeft')
    bottom_heatmap=heatmap(features,mode='bottomRight')
    corner=layer_heatmap_to_corner(tf.concat([top_heatmap,bottom_heatmap],axis=-1))
   
    
    # tloss=utils.focal_loss(top_heatmap,truth_top)
    # bloss=utils.focal_loss(bottom_heatmap,truth_bottom)
    # loss=tloss+bloss
    # accuracy,_=utils.get_accuracy(top_heatmap,bottom_heatmap,truth_points)

    # sess=tf.Session()
    # sess.run(tf.global_variables_initializer())


    # l=sess.run(loss)
    # acc=sess.run(accuracy)
    # print(l)
    # print(acc)