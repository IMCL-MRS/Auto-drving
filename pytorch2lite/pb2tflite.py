import tensorflow as tf

graph_def_file = "vae.pb"
input_arrays = ["input"]
output_arrays = ["output"]

converter = tf.lite.TocoConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)