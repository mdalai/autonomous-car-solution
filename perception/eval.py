
import tensorflow as tf
from tensorflow.python.framework import graph_util as tf_graph_util
import os.path
import os
import time
import scipy.misc
import numpy as np



def get_segmentation(sess, image, model_prediction_class,model_keep_prob,model_image_input ):
    start_time = timer()
    image = image[42:522,:,:]
    image = img_scale(image, 1/5, 1/5)

    predicted_class = sess.run( [model_prediction_class], {model_keep_prob: 1.0, model_image_input: [image]})
    binary_car =  (predicted_class[0][:,0]>0.5).reshape(image.shape[0], image.shape[1]).astype('uint8')
    binary_road = (predicted_class[0][:,1]>0.5).reshape(image.shape[0], image.shape[1]).astype('uint8')
    binary_car = img_scale(binary_car, 5, 5)
    binary_road = img_scale(binary_road, 5, 5)

    duration = timer() - start_time
    pred_time_ms = int(duration * 1000)

    return binary_car,binary_road, pred_time_ms

def f_score(gt,seg,beta):
    pred = np.sum(seg)
    if pred < 0.0001: pred = 0.0001
    tp = np.sum(seg*gt)
    fn = np.sum((seg==0)*(gt==1))
    if tp < 0.0001: tp = 0.0001
    if fn < 0.0001: fn = 0.0001
    p = tp/float(pred)     # precision
    r = tp/float(tp+fn)    # recall

    f_score = (1+beta**2)*((p*r)/(beta**2 * p + r))

    return f_score


def eval_nn(sess, batch_size, get_batches_fn, logits_op, loss_op, input_pl, keep_prob_pl,image_shape,label_pl):
    f_score_cars = []
    f_score_roads = []
    losses =[]
    for images, labels in get_batches_fn(batch_size):
        #print("Feature Shape: {}, Label Shape: {}.".format(images.shape,labels.shape))
        loss, logits = sess.run([loss_op, logits_op], 
            feed_dict = { input_pl: images, keep_prob_pl: 1.0, label_pl: labels })
        #logits = logits.reshape(images.shape[0],image_shape[0]*image_shape[1], 3)
        logits = logits.reshape(images.shape[0],image_shape[0]*image_shape[1], 2)
        #print("SHAPE logits: {}".format(logits.shape))
        for i,logit in enumerate(logits):
            #print("SHAPE logit: {}".format(logit.shape))
            binary_car =  (logit[:,0]>0.5).reshape(image_shape[0], image_shape[1]).astype('uint8')
            binary_road = (logit[:,1]>0.5).reshape(image_shape[0], image_shape[1]).astype('uint8')
            f_score_car = f_score(labels[i][:,:,0],binary_car,2.0)
            f_score_road = f_score(labels[i][:,:,1],binary_road,0.5)
            f_score_cars.append(f_score_car)
            f_score_roads.append(f_score_road)
        losses.append(loss)

    m_f_score_car = sum(f_score_cars)/float(len(f_score_cars))   
    m_f_score_road = sum(f_score_roads)/float(len(f_score_roads)) 
    m_f_score = (m_f_score_car + m_f_score_road)/2.0
    #print("EVAL: F_Car:{:.3f}, F_Road:{:.3f}, F_Avg:{:.3f}.".format(m_f_score_car,m_f_score_road,m_f_score) )
    #print("Val Loss:{}".format( sum(losses)/float(len(losses)) ))

    return m_f_score_car,m_f_score_road,m_f_score, sum(losses)/float(len(losses))

def weighted_loss(logits, labels, num_classes, head=None):
    """ median-frequency re-weighting """
    with tf.name_scope('loss'):
        #logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-10)
        logits = logits + epsilon

        # consturct one-hot label array
        #label_flat = tf.reshape(labels, (-1, 1))
        # should be [batch ,num_classes]
        #labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))
        labels = tf.to_float(labels)

        softmax = tf.nn.softmax(logits) + epsilon
        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), head), axis=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss

def loss(logits, labels, num_classes, head=None):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.upscore as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax = tf.nn.softmax(logits) + epsilon

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax),
                                           head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss


