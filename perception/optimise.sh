# first parameter $1 frozen_model_dir

# ideal transforms, but some of them break inference
#+add_default_attributes
#- remove_nodes(op=Identity, op=CheckNumerics)
#+fold_constants(ignore_errors=true)
#+fold_batch_norms
#+fold_old_batch_norms
#+fuse_resize_and_conv
#+quantize_weights increase time from 18ms to 50ms. quality seems ok
#- quantize_nodes
#+strip_unused_nodes
#+sort_by_execution_order

~/dev/tf/tensorflow-r1.3/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=$1/frozen_graph.pb \
--out_graph=$1/optimised_graph.pb \
--inputs=image_input \
--outputs=predictions/prediction_class \
--transforms='
add_default_attributes
fold_constants(ignore_errors=true)
fold_batch_norms
fold_old_batch_norms
fuse_resize_and_conv
quantize_weights
strip_unused_nodes
sort_by_execution_order'