import numpy as np
import os
import tensorflow as tf
import cv2
def img_scale(img, fx, fy):
    return cv2.resize(img, None, fx=fx, fy=fy)
class Segmentor(object):
    def __init__(self):       
        pwd = os.path.dirname(os.path.realpath(__file__))
        PATH_TO_MODEL = os.path.join(pwd,'optimized_graph13.pb')		
        self.cut_off_part1 = np.zeros(shape=(42,800)).astype('uint8')
        self.cut_off_part2 = np.zeros(shape=(78,800)).astype('uint8')      
        graph = tf.Graph()
        config = tf.ConfigProto()
        jit_level = tf.OptimizerOptions.ON_2
        config.graph_options.optimizer_options.global_jit_level = jit_level
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_input = graph.get_tensor_by_name('image_input:0')
            self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
            self.pred_class = graph.get_tensor_by_name('predictions/prediction_class:0')

        self.sess = tf.Session(graph=graph, config=config)

    def get_segmentation_single(self, image):
        #image = scipy.misc.imread(img)[170:522,:,:]
        image = image[42:522,:,:]
        image = img_scale(image, 1/5, 1/5)
        im_result = self.sess.run(
                [self.pred_class],
                {self.keep_prob: 1.0, self.image_input: [image]})    
        binary_car =  (im_result[0][:,0]>0.5).reshape(image.shape[0], image.shape[1]).astype('uint8')
        binary_road = (im_result[0][:,1]>0.5).reshape(image.shape[0], image.shape[1]).astype('uint8')
        binary_car = img_scale(binary_car, 5, 5)
        binary_road = img_scale(binary_road, 5, 5)
        binary_car =  np.append(np.append(self.cut_off_part1,binary_car,axis=0),self.cut_off_part2,axis=0)
        binary_road = np.append(np.append(self.cut_off_part1,binary_road,axis=0),self.cut_off_part2,axis=0)            
            
        return binary_car,binary_road

    def get_segmentation_multi(self, images):
        #image = scipy.misc.imread(img)[170:522,:,:]
        images = images[:,42:522,:,:]
        images = np.array([img_scale(image, 1/5, 1/5) for image in images])
        #print(images.shape)
        im_results = self.sess.run(
                [self.pred_class],
                {self.keep_prob: 1.0, self.image_input: images})  
        #print(im_results[0].shape)
        binary_cars =  (im_results[0][:,0]>0.5).reshape(images.shape[0],images.shape[1],images.shape[2]).astype('uint8')
        binary_roads = (im_results[0][:,1]>0.5).reshape(images.shape[0],images.shape[1],images.shape[2]).astype('uint8')
        binary_cars = [img_scale(binary_car, 5, 5) for binary_car in binary_cars]
        binary_roads = [img_scale(binary_road, 5, 5) for binary_road in binary_roads]
        binary_cars =  [np.append(np.append(self.cut_off_part1,binary_car,axis=0),self.cut_off_part2,axis=0) for binary_car in binary_cars]
        binary_roads = [np.append(np.append(self.cut_off_part1,binary_road,axis=0),self.cut_off_part2,axis=0) for binary_road in binary_roads]         
            
        return binary_cars,binary_roads