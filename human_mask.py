import os
import tensorflow as tf

trained_checkpoint_prefix = '/media/vasile/W10/JPPNet-s2/model.ckpt-205632'
export_dir = os.path.join(os.path.dirname(trained_checkpoint_prefix), '1')

#loaded_graph = tf.Graph()
#with tf.compat.v1.Session(graph=loaded_graph) as sess:
#    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
#    loader.restore(sess, trained_checkpoint_prefix)
#    tf.compat.v1.enable_resource_variables()
#    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
#    builder.add_meta_graph_and_variables(sess, ["train", "serve"], strip_default_attrs=True)
#    builder.save()
jpp_model = tf.saved_model.load(export_dir)
try:
    print(jpp_model.signatures["serve"].summary())
except Exception as e:
    print(e)
print([i for i in jpp_model.signatures.keys()])
print([i for i in jpp_model.signatures.items()])
print([i for i in jpp_model.signatures.values()])
