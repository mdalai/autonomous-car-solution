
import tensorflow as tf
from tensorflow.python.framework import graph_util as tf_graph_util
import os.path
import os
import warnings
import shutil
import time
import argparse
from timeit import default_timer as timer
import math

from tqdm import tqdm
import scipy.misc
import numpy as np
#from moviepy.editor import VideoFileClip
import helper

def session_config():
    # tensorflow GPU config
    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU': 1})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # playing with JIT level, this can be set to ON_1 or ON_2
    #jit_level = tf.OptimizerOptions.ON_1 # this works on Ubuntu tf1.3 but does not improve performance
    jit_level = tf.OptimizerOptions.ON_2
    config.graph_options.optimizer_options.global_jit_level = jit_level

def save_model(sess, model_dir):
    TAG = 'FCN8'
    builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
    builder.add_meta_graph_and_variables(sess, [TAG])  # tag
    builder.save()

def load_model(sess, model_dir):
    """ load trained model using SavedModelBuilder. can only be used for inference """
    TAG = 'FCN8'
    # tf.reset_default_graph()
    # sess.run(tf.global_variables_initializer())
    tf.saved_model.loader.load(sess, [TAG], model_dir)
    # we need to re-assign the following ops to instance variables for prediction
    # we cannot continue training from this state as other instance variables are undefined
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name("image_input:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    prediction_class = graph.get_tensor_by_name("predictions/prediction_class:0")

    return image_input,keep_prob,prediction_class

# added it for speeding up the inference. No need to call Graph hopefully.
def create_predictions(logits):
    """ define prediction probabilities and classes """
    with tf.name_scope("predictions"):
        # this is how you give a name to a tf op
        #logits = tf.identity(output, name='logits')
        prediction_softmax = tf.nn.softmax(logits, name="prediction_softmax")
        prediction_class = tf.cast(tf.greater(prediction_softmax, 0.5), dtype=tf.float32, name='prediction_class')
        print("MYPRINT predict Class: {}".format(prediction_class))
        #prediction_class_idx = tf.cast(tf.argmax(prediction_class, axis=1), dtype=tf.uint8, name='prediction_class_idx')
        #tf.summary.image('prediction_class_idx', tf.expand_dims(tf.div(tf.cast(prediction_class_idx, dtype=tf.float32), float(self._num_classes)), -1), max_outputs=2)

    return prediction_softmax, prediction_class

def freeze_graph(args):
    # based on https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    if args.ckpt_dir is None:
        print("for freezing need --ckpt_dir")
        return
    if args.frozen_model_dir is None:
        print("for freezing need --frozen_model_dir")
        return

    checkpoint = tf.train.get_checkpoint_state(args.ckpt_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    print("freezing from {}".format(input_checkpoint))
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    print("{} ops in the input graph".format(len(input_graph_def.node)))

    output_node_names = "predictions/prediction_class"

    # freeze graph
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # use a built-in TF helper to export variables to constants
        output_graph_def = tf_graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")
        )

    print("{} ops in the frozen graph".format(len(output_graph_def.node)))

    # if the directory already exists, delete the Dir, subDir and files in it.
    if os.path.exists(args.frozen_model_dir):
        shutil.rmtree(args.frozen_model_dir)

    # save model in same format as usual
    print('saving frozen model as saved_model to {}'.format(args.frozen_model_dir))
    tf.reset_default_graph()
    tf.import_graph_def(output_graph_def, name='')
    with tf.Session() as sess:
        save_model(sess, args.frozen_model_dir)

    print('saving frozen model as frozen_graph.pb (for transforms) to {}'.format(args.frozen_model_dir))
    with tf.gfile.GFile(args.frozen_model_dir+'/frozen_graph.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())


def save_optimised_graph(args):
    """ optimize frozen graph for inference """
    if args.frozen_model_dir is None:
        print("for optimise need --frozen_model_dir")
        return


    # reading optimised graph
    tf.reset_default_graph()
    gd = tf.GraphDef()
    output_graph_file = args.frozen_model_dir+"/optimised_graph2.pb"
    with tf.gfile.Open(output_graph_file, 'rb') as f:
        gd.ParseFromString(f.read())
    tf.import_graph_def(gd, name='')
    print("{} ops in the optimised graph".format(len(gd.node)))

    # save model in same format as usual
    #shutil.rmtree(args.optimised_model_dir, ignore_errors=True)
    #if not os.path.exists(args.optimised_model_dir):
    #    os.makedirs(args.optimised_model_dir)

    print('saving optimised model as saved_model to {}'.format(args.optimised_model_dir))
    tf.reset_default_graph()
    tf.import_graph_def(gd, name='')
    with tf.Session() as sess:
        save_model(sess, args.optimised_model_dir+'/savedmodel')
    #shutil.move(args.frozen_model_dir+'/optimised_graph.pb', args.optimised_model_dir)


def predict_image(sess, image, model_prediction_class,model_keep_prob,model_image_input ):
    # this image size is arbitrary and may break middle of decoder in the network.
    # need to feed FCN images sizes in multiples of 32

    # run TF prediction
    start_time = timer()
    predicted_class = sess.run( [model_prediction_class], {model_keep_prob: 1.0, model_image_input: [image]})
    predicted_class = predicted_class[0]
    predicted_class = np.array(predicted_class, dtype=np.uint8)
    duration = timer() - start_time
    tf_time_ms = int(duration * 1000)

    # overlay on image
    start_time = timer()
    result_im = scipy.misc.toimage(image)

    segmentation_car  = predicted_class[:,0].reshape(image.shape[0], image.shape[1], 1)
    segmentation_road = predicted_class[:,1].reshape(image.shape[0], image.shape[1], 1)
    mask_car = np.dot(segmentation_car, np.array([[255, 0, 0, 127]]))
    mask_car = scipy.misc.toimage(mask_car, mode="RGBA")
    mask_road = np.dot(segmentation_road, np.array([[0, 255, 0, 127]]))
    mask_road = scipy.misc.toimage(mask_road, mode="RGBA")
    result_im.paste(mask_car, box=None, mask=mask_car)
    result_im.paste(mask_road, box=None, mask=mask_road)

    segmented_image = np.array(result_im)
    duration = timer() - start_time
    img_time_ms = int(duration * 1000)

    return segmented_image, tf_time_ms, img_time_ms

def predict_video(args, image_shape=None):
    if args.video_file_in is None:
        print("for video processing need --video_file_in")
        return
    if args.video_file_out is None:
        print("for video processing need --video_file_out")
        return

    def process_frame(image):
        image = scipy.misc.imread(image)[170:522,:,:]
        if image_shape is not None:
            image = scipy.misc.imresize(image, image_shape)
        segmented_image, tf_time_ms, img_time_ms = predict_image(sess, image, \
                                prediction_class,keep_prob,image_input)
        return segmented_image

    tf.reset_default_graph()
    with tf.Session(config=session_config()) as sess:
        image_input,keep_prob,prediction_class = load_model(sess, 'trained_model' if args.model_dir is None else args.model_dir)
        print('Running on video {}, output to: {}'.format(args.video_file_in, args.video_file_out))

        input_clip = VideoFileClip(args.video_file_in)
        annotated_clip = input_clip.fl_image(process_frame)
        annotated_clip.write_videofile(video_file_out, audio=False)
        # for half size
        # ubuntu/1080ti. with GPU ??fps. with CPU the same??
        # mac/cpu 1.8s/frame
        # full size 1280x720
        # ubuntu/gpu 1.2s/frame i.e. 0.8fps :(
        # ubuntu/cpu 1.2fps
        # mac cpu 6.5sec/frame




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',
                        help='what to do: predict/freeze/optimise/video',
                        type=str,
                        choices=['predict', 'freeze', 'optimise', 'video'])
    #parser.add_argument('-g', '--gpu', help='number of GPUs to use. default 0 (use CPU)', type=int, default=0)
    #parser.add_argument('-gm','--gpu_mem', help='GPU memory fraction to use. default 0.9', type=float, default=0.9)
    #parser.add_argument('-x','--xla', help='XLA JIT level. default None', type=int, default=None, choices=[1,2])
    #parser.add_argument('-ep', '--epochs', help='training epochs. default 0', type=int, default=0)
    #parser.add_argument('-bs', '--batch_size', help='training batch size. default 5', type=int, default=5)
    #parser.add_argument('-lr', '--learning_rate', help='training learning rate. default 0.0001', type=float, default=0.0001)
    #parser.add_argument('-kp', '--keep_prob', help='training dropout keep probability. default 0.9', type=float, default=0.9)
    #parser.add_argument('-rd', '--runs_dir', help='training runs directory. default runs', type=str, default='runs')
    parser.add_argument('-cd', '--ckpt_dir', help='training checkpoints directory. default ckpt', type=str, default='ckpt')
    #parser.add_argument('-sd', '--summary_dir', help='training tensorboard summaries directory. default summaries', type=str, default='summaries')
    #parser.add_argument('-md', '--model_dir', help='model directory. default None - model directory is created in runs. needed for predict', type=str, default=None)
    parser.add_argument('-fd', '--frozen_model_dir', help='model directory for frozen graph. for freeze', type=str, default=None)
    parser.add_argument('-od', '--optimised_model_dir', help='model directory for optimised graph. for optimize', type=str, default=None)
    parser.add_argument('-ip', '--images_paths', help="images path/file pattern. e.g. 'train/img*.png'", type=str, default=None)
    #parser.add_argument('-lp', '--labels_paths', help="label images path/file pattern. e.g. 'train/label*.png'", type=str, default=None)
    parser.add_argument('-vi', '--video_file_in', help="mp4 video file to process", type=str, default=None)
    parser.add_argument('-vo', '--video_file_out', help="mp4 video file to save results", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # fix plugin for moviepy
    #import imageio
    #imageio.plugins.ffmpeg.download()

    print("action={}".format(args.action))
    if args.action=='predict':
        print('images_paths={}'.format(args.images_paths))
        image_shape = (256, 512)
        predict_files(args, image_shape)
    elif args.action == 'freeze':
        freeze_graph(args)
    elif args.action == 'optimise':
        save_optimised_graph(args)
    elif args.action == 'video':
        #image_shape = None
        image_shape = (int(720/2), int(1280/2))
        predict_video(args, image_shape)

