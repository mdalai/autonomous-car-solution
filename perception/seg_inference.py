import numpy as np
import os
import sys
import tensorflow as tf
from collections import defaultdict
import time
import scipy.misc


class Segmentor(object):
    def __init__(self):       
        pwd = os.path.dirname(os.path.realpath(__file__))
        PATH_TO_MODEL = os.path.join(pwd,'optimized_graph.pb')
		
        self.cut_off_part1 = (np.zeros(shape=(170,800))==1).astype('uint8')
        self.cut_off_part2 = (np.zeros(shape=(78,800))==1).astype('uint8')
        
        # load a FROZEN TF model into memory        
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        # JIT level, this can be set to ON_1 or ON_2 
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level
        #config.gpu_options.allow_growth = True
        
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_input = self.graph.get_tensor_by_name('image_input:0')
            self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
            self.logits = self.graph.get_tensor_by_name('logits_2d:0')

        self.sess = tf.Session(graph=self.graph, config=config)


    def get_segmentation(self, image):
        with self.graph.as_default():
            #image = scipy.misc.imread(img)[:544,:,:]
            image = image[:544,:,:]
            im_softmax = self.sess.run(
                [tf.nn.softmax(self.logits)],
                {self.keep_prob: 1.0, self.image_input: [image]})    

            binary_car =  (im_softmax[0][:,0]>0.5).reshape(image.shape[0], image.shape[1]).astype('uint8')
            binary_road = (im_softmax[0][:,1]>0.5).reshape(image.shape[0], image.shape[1]).astype('uint8')

            binary_car =  np.append(np.append(self.cut_off_part1,binary_car,axis=0),self.cut_off_part2,axis=0)
            binary_road = np.append(np.append(self.cut_off_part1,binary_road,axis=0),self.cut_off_part2,axis=0)            
            
        return binary_car,binary_road