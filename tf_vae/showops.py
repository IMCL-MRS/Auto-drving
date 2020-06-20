import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format

import tensorflow as tf
from argparse import ArgumentParser

def convert_pb_to_pbtxt(filename):
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()

        graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name='')

        tf.train.write_graph(graph_def, './', 'protobuf.pbtxt', as_text=True)
    return


def convert_pbtxt_to_pb(filename):
    """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.
    Args:
      filename: The name of a file containing a GraphDef pbtxt (text-formatted
        `tf.GraphDef` protocol buffer data).
    """
    with tf.gfile.FastGFile(filename, 'r') as f:
        graph_def = tf.GraphDef()

        file_content = f.read()

        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, './', 'protobuf.pb', as_text=False)


def main():
    # python3 showops.py --checkpoint vae_40/vae-0.data-00000-of-00001 --model vae-0.meta --out-path=./
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)
    parser.add_argument('--model', type=str,
                        dest='model',
                        help='.meta for your model',
                        metavar='MODEL', required=True)
    parser.add_argument('--out-path', type=str,
                        dest='out_path',
                        help='model output directory',
                        metavar='MODEL_OUT', required=True)
    opts = parser.parse_args()
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(opts.model)
    builder = tf.saved_model.builder.SavedModelBuilder(opts.out_path)
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, opts.checkpoint)
        print("Model restored.")
        builder.add_meta_graph_and_variables(sess,
                                       ['tfckpt2pb'],
                                       strip_default_attrs=False)
        builder.save()

if __name__ == '__main__':
    main()

