import tensorflow as tf
import argparse
"""
convert_ckpt_to_graph.py --meta_graph_file 'data/model1/model1.meta'  --ckpt_dir './data/model1' --export_dir './savedmodel'
"""

# command line arguments
parser = argparse.ArgumentParser(
    description='Convert a checkpoint to frozen graph')
parser.add_argument(
    '--meta_graph_file',
    type=str,
    default="model.meta",
    help='The meta graph file.')
parser.add_argument(
    '--ckpt_dir',
    type=str,
    default="model/",
    help='The checkpoint folder to be converted')
parser.add_argument(
    '--export_dir',
    type=str,
    default="./savedmodel",
    help='Output dir.')

args = parser.parse_args()


# initialise the saver
saver = tf.train.import_meta_graph(args.meta_graph_file)

with tf.Session() as sess:
    # restore all variables from checkpoint
    saver.restore(sess, tf.train.latest_checkpoint(args.ckpt_dir))

    graph = tf.get_default_graph()
    #labels = graph.get_tensor_by_name("labels:0")
    logits_2d = graph.get_tensor_by_name("logits_2d:0")
    #labels_2d = graph.get_tensor_by_name("labels_2d:0")
    #loss = graph.get_tensor_by_name("loss:0")
    image_input = graph.get_tensor_by_name("image_input:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    ############## save the model ######################################
    export_path =  './savedmodel'
    builder = tf.saved_model.builder.SavedModelBuilder(args.export_dir)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(image_input)
    tensor_info_kp = tf.saved_model.utils.build_tensor_info(keep_prob)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(logits_2d)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x, 'keep_prob': tensor_info_kp},
            outputs={'logits': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature 
        },
        legacy_init_op=legacy_init_op)

    builder.save()

    print("Done exporting! Check folder:{} ".format(args.export_dir))