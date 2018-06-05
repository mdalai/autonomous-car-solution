import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import matplotlib.image as mpimg
import cv2

############## DATA PRE-Processing BEGIN ################################
# Vehicles - class 10
CAR_COLOR = np.array([10, 0, 0])
# Roads - class 7
ROAD_COLOR = np.array([7, 0, 0])
# RoadLines - class 6
ROADLINE_COLOR = np.array([6, 0, 0])

def data_preprocess(img_path, label_path):
    train_img = scipy.misc.imread(img_path)
    #train_img = mpimg.imread(img_path)
    label_img = scipy.misc.imread(label_path)
    #train_img = train_img[170:522,:,:]
    #label_img = label_img[170:522,:,:]
    #train_img = train_img[138:522,:,:]
    #label_img = label_img[138:522,:,:]
    train_img = train_img[42:522,:,:]
    label_img = label_img[42:522,:,:]

    # label processing
    gt1 = np.all((label_img == ROAD_COLOR) | (label_img == ROADLINE_COLOR), axis=2)
    gt2 = np.append(np.all(label_img[:452,:,:] == CAR_COLOR, axis=2), \
                    np.all(label_img[452:,:,:] == np.array([222, 0, 0]), axis=2),axis=0)
    gt3 = gt1 == gt2
    gt1 = gt1.reshape(*gt1.shape,1)
    gt2 = gt2.reshape(*gt2.shape,1)
    gt3 = gt3.reshape(*gt3.shape,1)
    gt_label = np.concatenate((gt2, gt1, gt3), axis=2)
    
    return train_img, gt_label.astype('uint8')  

def translate(img,x,y):    
    M = np.float32([[1,0,x],[0,1,y]])
    rows,cols,_ = img.shape
    return cv2.warpAffine(img,M,(cols,rows),borderMode=cv2.BORDER_REFLECT)

def brightness(img):
    x= random.randint(-150,150)
    table = np.array([i+x for i in np.arange(0, 256)])
    table[table<0]=0
    table[table>255]=255
    table=table.astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)


def img_scale(img, fx, fy):
    return cv2.resize(img, None, fx=fx, fy=fy)

def data_preprocess2(img_path, label_path):
    train_img = scipy.misc.imread(img_path)
    label_img = scipy.misc.imread(label_path)
    train_img = train_img[42:522,:,:]
    label_img = label_img[42:522,:,:]

    # label processing
    gt2 = np.append(np.all(label_img[:452,:,:] == CAR_COLOR, axis=2), \
                    np.all(label_img[452:,:,:] == np.array([222, 0, 0]), axis=2),axis=0)
    gt3 = np.logical_not(gt2)
    gt2 = gt2.reshape(*gt2.shape,1)
    gt3 = gt3.reshape(*gt3.shape,1)
    gt_label = np.concatenate((gt2, gt3), axis=2)
    
    return train_img, gt_label.astype('uint8')
############## DATA PRE-Processing END #################################

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def gen_batch_function2(data_folder):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :return:
    """
    def get_batches_fn2(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
        label_paths = glob(os.path.join(data_folder, 'CameraSeg', '*.png'))

        # pair image_paths and label_paths before shuffle
        paths = list(zip(image_paths,label_paths))
        random.shuffle(paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_labels = []
            for image_file,gt_image_file in paths[batch_i:batch_i+batch_size]:
                # data preprocessing
                image,gt_label = data_preprocess2(image_file, gt_image_file)
                image = img_scale(image,1/5,1/5)
                gt_label = img_scale(gt_label,1/5,1/5)

                images.append(image)
                gt_labels.append(gt_label)

                """
                for i in range(3):
                    # Brightness
                    image = brightness(image)

                    images.append(image)
                    gt_labels.append(gt_label)

                    if len(images) >= batch_size:
                        yield np.array(images), np.array(gt_labels)
                        images = []
                        gt_labels = []
                """

                
                # Flip image
                #images.append(image[:,::-1,:])
                #gt_labels.append(gt_label[:,::-1,:])

                """
                # Augmentation
                for i in range(3):
                    # Random translate x,y
                    x = random.randint(-100,100)
                    y = random.randint(-30,30)
                    image = translate(image,x,y)
                    gt_label = translate(gt_label,x,y)
                    # Brightness
                    image = brightness(image)

                    images.append(image)
                    gt_labels.append(gt_label)
                    
                    # Flip image
                    images.append(image[:,::-1,:])
                    gt_labels.append(gt_label[:,::-1,:])

                if len(images) >= batch_size:
                    yield np.array(images), np.array(gt_labels)
                    images = []
                    gt_labels = []

                """


            yield np.array(images), np.array(gt_labels)
    return get_batches_fn2

def val_gen_batch_function(data_folder):
    def val_get_batches_fn(batch_size):
        image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
        label_paths = glob(os.path.join(data_folder, 'CameraSeg', '*.png'))

        # pair image_paths and label_paths before shuffle
        paths = list(zip(image_paths,label_paths))
        random.shuffle(paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_labels = []
            for image_file,gt_image_file in paths[batch_i:batch_i+batch_size]:
                # data preprocessing
                image,gt_label = data_preprocess2(image_file, gt_image_file)
                image = img_scale(image,1/5,1/5)
                gt_label = img_scale(gt_label,1/5,1/5)
                images.append(image)
                gt_labels.append(gt_label)
            yield np.array(images), np.array(gt_labels)
    return val_get_batches_fn


def gen_test_output2(sess, logits, keep_prob, image_pl, data_folder):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, '*.png')):
        image = scipy.misc.imread(image_file)[42:522,:,:]
        #image = scipy.misc.imread(image_file)[170:522,:,:]
        image = img_scale(image,1/5,1/5)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax_car = im_softmax[0][:, 0].reshape(image.shape[0], image.shape[1])
        im_softmax_road = im_softmax[0][:, 1].reshape(image.shape[0], image.shape[1])
        segmentation_car = (im_softmax_car > 0.5).reshape(image.shape[0], image.shape[1], 1)
        segmentation_road = (im_softmax_road > 0.5).reshape(image.shape[0], image.shape[1], 1)
        mask_car = np.dot(segmentation_car, np.array([[255, 0, 0, 127]]))
        mask_car = scipy.misc.toimage(mask_car, mode="RGBA")
        mask_road = np.dot(segmentation_road, np.array([[0, 255, 0, 127]]))
        mask_road = scipy.misc.toimage(mask_road, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask_car, box=None, mask=mask_car)
        street_im.paste(mask_road, box=None, mask=mask_road)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output2(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'test'))
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
