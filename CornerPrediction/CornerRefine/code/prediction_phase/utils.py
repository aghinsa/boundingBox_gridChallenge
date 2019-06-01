import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tf_layers

def corner_pool(inputs,kernel_size=[3,3],mode='topLeft'):
    if mode=='bottomRight':
        pad_bottom=np.array([[0,0],[0,kernel_size[0]-1],[0,0],[0,0]])
        pad_right=np.array([[0,0],[0,0],[0,kernel_size[1]-1],[0,0]])
        temp_bottom=tf.pad(inputs,paddings=pad_bottom)
        temp_right=tf.pad(inputs,paddings=pad_right)
        temp_bottom=tf.layers.max_pooling2d(temp_bottom, (kernel_size[0],1), (1,1), padding='valid')
        temp_right=tf.layers.max_pooling2d(temp_right, (1,kernel_size[1]), (1,1), padding='valid')
        outputs=tf.add(temp_bottom,temp_right)
    elif mode=='topLeft':
        pad_top=np.array([[0,0],[kernel_size[0]-1,0],[0,0],[0,0]])
        pad_left=np.array([[0,0],[0,0],[kernel_size[1]-1,0],[0,0]])
        temp_top=tf.pad(inputs,paddings=pad_top)
        temp_left=tf.pad(inputs,paddings=pad_left)
        temp_top=tf.layers.max_pooling2d(temp_top, (kernel_size[0],1), (1,1), padding='valid')
        temp_left=tf.layers.max_pooling2d(temp_left, (1,kernel_size[1]), (1,1), padding='valid')
        outputs=tf.add(temp_top,temp_left)
    return outputs

def upsample(x,n_channels,kernel=3,stride=2,
        activation_fn=tf.nn.leaky_relu,
        normalizer_fn=tf_layers.batch_norm):
    """
    x is encoded
    """
    h_new = (x.get_shape()[1].value) * stride
    w_new = (x.get_shape()[2].value) * stride
    up = tf.image.resize_nearest_neighbor(x, [h_new, w_new])
    result=tf_layers.conv2d(up,num_outputs=n_channels,kernel_size=kernel,
        stride=1,normalizer_fn=normalizer_fn,activation_fn=activation_fn)
    return result


def get_predictions_from_heatmap(heatmap,img_size):
    """
    heatmap=[?,h,w]
    return preds =[?,2] x.y
    """
    mask=tf.contrib.layers.flatten(heatmap)
    idx=tf.argmax(mask,axis=1)
    ridx=tf.math.floor(idx/img_size[1])
    cidx=tf.mod(idx,img_size[0])
    ridx=tf.cast(ridx,tf.int32)
    cidx=tf.cast(cidx,tf.int32)

    preds=tf.stack([ridx,cidx],axis=1)
    return preds



def tf_box_mean_iou(preds,labels,mean=True):
    """
    preds =? x1,y1,x2,y2
    """
    preds=preds[:,0:4]
    x1=tf.maximum(preds[:,0],labels[:,0])
    y1=tf.maximum(preds[:,1],labels[:,1])
    x2=tf.minimum(preds[:,2],labels[:,2])
    y2=tf.minimum(preds[:,3],labels[:,3])

    ia=tf.maximum(x2-x1,tf.zeros_like(x2))*tf.maximum(y2-y1,tf.zeros_like(y2))
    b1a=(preds[:,2]-preds[:,0])*(preds[:,3]-preds[:,1])
    b2a=(labels[:,2]-labels[:,0])*(labels[:,3]-labels[:,1])
    iou=tf.divide(ia,(b1a+b2a-ia))
    if mean:
        iou=tf.reduce_mean(iou)
    return iou

def get_accuracy(pred_top,pred_bottom,true_points,img_size=(512,512)):
    """
    preds[?,h,w,1]
    true_points=[?,4][x1,y1,x2,y2]
    """
    pred_top=tf.cast(pred_top,tf.float32)
    pred_bottom=tf.cast(pred_bottom,tf.float32)
    true_points=tf.cast(true_points,tf.float32)


    pred_top=tf.squeeze(pred_top,axis=-1)
    pred_bottom=tf.squeeze(pred_bottom,axis=-1)
   

    pred_top_corner=get_predictions_from_heatmap(pred_top,img_size)
    pred_bottom_corner=get_predictions_from_heatmap(pred_bottom,img_size)

    pred_points=tf.concat([pred_top_corner,pred_bottom_corner],axis=-1)
    pred_points=tf.cast(pred_points,tf.float32)

    acc=tf_box_mean_iou(pred_points,true_points)
    
    
    return acc,pred_points

