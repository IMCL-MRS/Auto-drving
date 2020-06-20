import tensorflow as tf
from tensorflow.python.saved_model import tag_constants,signature_constants

x = tf.placeholder(tf.float32, [1, 120, 200, 1])  # Placeholder:0
y = tf.placeholder(tf.float32, [1, 120, 200, 1])  # Placeholder_1:0
inputs = {'input': tf.saved_model.utils.build_tensor_info(x)}
outputs = {'output': tf.saved_model.utils.build_tensor_info(y)}
signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs)
signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}

converter = tf.contrib.lite.TFLiteConverter.from_saved_model("save_model/1",signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open("vae.tflite", "wb").write(tflite_quant_model)