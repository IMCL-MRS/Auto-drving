import os
import tensorflow as tf
import numpy as np
from PIL import Image
import glob as glob
from tensorflow.python.saved_model import tag_constants,signature_constants
# from tensorflow.python.client import graph_util
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

#
# def func_ck2pb():
#     trained_checkpoint_prefix = 'vae_40/vae-0'
#     export_dir = os.path.join('save_model', '1')
#
#     graph = tf.Graph()
#     with tf.Session(graph=graph) as sess:
#         # Restore from checkpoint
#         loader = tf.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
#         loader.restore(sess, trained_checkpoint_prefix)
#
#         # Export checkpoint to SavedModel
#         builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
#         x = tf.placeholder(tf.float32, [1, 120, 200, 1])
#         y = tf.placeholder(tf.float32, [1, 120, 200, 1])
#         inputs = {'input': tf.saved_model.utils.build_tensor_info(x)}
#         outputs = {'output': tf.saved_model.utils.build_tensor_info(y)}
#         signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs)
#         signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
#         # builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING] )
#         # builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], {'test_signature': None})
#         # builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], strip_default_attrs=True)
#         # builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], signature_def_map={'vae_image':signature})
#         builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], signature_def_map=signature_map)
#         builder.save()
#         print("covnert from ck2 to saved pb model successfully!")
#
#
# def func_pb2tflite():
#     # x = tf.placeholder(tf.float32, [1, 120, 200, 1])  # Placeholder:0
#     # y = tf.placeholder(tf.float32, [1, 120, 200, 1])  # Placeholder_1:0
#     # inputs = {'input': tf.saved_model.utils.build_tensor_info(x)}
#     # outputs = {'output': tf.saved_model.utils.build_tensor_info(y)}
#     # signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs)
#     # signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
#
#     converter = tf.contrib.lite.TFLiteConverter.from_saved_model("save_model/1",signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
#     # converter.optimizations = [tf.contrib.lite.TFLiteConverter.OPTIMIZE_FOR_SIZE]
#     tflite_quant_model = converter.convert()
#     open("vae.tflite", "wb").write(tflite_quant_model)
#
#     converter.post_training_quantize = True
#     tflite_quantized_model = converter.convert()
#     open("quantized_vae.tflite", "wb").write(tflite_quantized_model)
#     print("covnert from pb to tflite model successfully!")

def dump_graph_nodes(graph):
    fd = open("node_list.txt", "a+")
    for op in graph.get_operations():
        print(op.name)
        fd.write(op.name + "\r\n")

def func_ck2_freezepb(input_checkpoint, output_graph):
    output_node_names = "output/Sigmoid"  # 获取的节点
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    dump_graph_nodes(graph)
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        # print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


def freeze_pb_tflite():
    # convert = tf.contrib.lite.TFLiteConverter.from_frozen_graph("vae_frozen.pb", input_arrays=["input"], output_arrays=["output/Sigmoid"])
    # convert = tf.lite.TFLiteConverter.from_frozen_graph("vae_frozen.pb", input_arrays=["input"], output_arrays=["enc_fc_mu/BiasAdd"], input_shapes={"input": [1, 120, 200, 1]})
    convert = tf.lite.TFLiteConverter.from_frozen_graph("vae_frozen.pb", input_arrays=["enc_fc_mu/BiasAdd"], output_arrays=["output/Sigmoid"])
    # convert.post_training_quantize = True

    # train = []
    # file_list = glob.glob("./train/" + "*.png")
    # for i in range(100):
    #     image_name = file_list[i]
    #     image = Image.open(image_name)
    #     image = image.convert('L').resize((200,120))
    #     image = np.array(image).astype(np.float32)
    #     image = tf.convert_to_tensor(image)
    #     train.append(image)
    #
    # # train = tf.convert_to_tensor(np.array(train, dtype='float32'))
    # my_ds = tf.data.Dataset.from_tensor_slices((train)).batch(1)


    # test_data = np.random.normal(0,1,(1, 120,200,1)).astype(np.float32)
    # # POST TRAINING QUANTIZATION
    # def representative_dataset_gen():
    #     for input_value in range(10):
    #         yield [test_data]
    #
    # convert.representative_dataset = representative_dataset_gen

    # convert.optimizations = [tf.lite.Optimize.DEFAULT]
    # convert.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # convert.inference_input_type = tf.int8
    # convert.inference_output_type = tf.int8
    # convert.allow_custom_ops = True
    tflite_model = convert.convert()
    open("vae_quantized.tflite", "wb").write(tflite_model)

    # 当需要给定输入数据形式时，给出输入格式：
    # convert = tf.lite.TFLiteConverter.from_frozen_graph("vae_frozen.pb", input_arrays=["input"],
    #                                                     output_arrays=["output/sigmoid"],
    #                                                     input_shapes={"input": [1, 120, 200, 1]})
    # convert.post_training_quantize = True
    # tflite_model = convert.convert()
    # open(path + "quantized_model.tflite", "wb").write(tflite_model)
    # print("finish!")

def func_freezepb2_tflite_fp32(out_name):
    # 1. encoder
    convert = tf.contrib.lite.TFLiteConverter.from_frozen_graph("vae_frozen.pb", input_arrays=["input"], output_arrays=["enc_fc_mu/BiasAdd"], input_shapes={"input": [1, 120, 200, 1]})
    # 2. decoder
    # convert = tf.lite.TFLiteConverter.from_frozen_graph("vae_frozen.pb", input_arrays=["enc_fc_mu/BiasAdd"], output_arrays=["output/Sigmoid"])
    tflite_model = convert.convert()
    open(out_name, "wb").write(tflite_model)

def func_freezepb2_tflite_int8(out_name):
    # 1. encoder
    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph("vae_frozen.pb", input_arrays=["input"], output_arrays=["enc_fc_mu/BiasAdd"], input_shapes={"input": [1, 120, 200, 1]})
    converter.experimental_new_converter = True
    converter.optimizations = [tf.contrib.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_dataset_gen_321
    tflite_model = converter.convert()
    open(out_name, "wb").write(tflite_model)

if __name__ == "__main__":
    model_path = "./test_model/vae-0"
    func_ck2_freezepb(model_path, "vae_frozen.pb")
    func_freezepb2_tflite_fp32("encode.tflite")


