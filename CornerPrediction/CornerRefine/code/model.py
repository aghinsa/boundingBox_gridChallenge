net=dense(inputs,num_outputs=32)
    net=dense(inputs,num_outputs=64)
    net1=dense(net,num_outputs=256)

    net=dense(net1,num_outputs=512)
    net=dense(net,num_outputs=1024)
    net=dense(net,num_outputs=512)
    net=tf.concat([net,net1],axis=-1)
    net=dense(net,num_outputs=256)
    net=dense(net,num_outputs=64)
    net=dense(net,num_outputs=32)
    net=dense(net,num_outputs=8)
    net=dense(net,num_outputs=4,activation_fn=None)