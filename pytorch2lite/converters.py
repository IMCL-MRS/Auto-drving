import shutil
from pathlib import Path
import onnx
import tensorflow as tf
from tensorflow.python.keras import backend as K
from onnx2keras import onnx_to_keras


def pytorch2savedmodel(onnx_model_path, saved_model_dir):
    #onnx_model = onnx.load(onnx_model_path)
    onnx_model = onnx.load_model(onnx_model_path)

    # input_names = ['input']
    input_names = ['input1','input2']

    # k_model = onnx_to_keras(onnx_model=onnx_model, change_ordering=True, verbose=False)
    #k_model = onnx_to_keras(onnx_model=onnx_model, input_names=input_names, change_ordering=True, verbose=False)
    print("#"*10)
    k_model = onnx_to_keras(onnx_model, ['input1','input2'], change_ordering=False, verbose=True)
    print("#"*10)

    weights = k_model.get_weights()

    K.set_learning_phase(0)

    saved_model_dir = Path(saved_model_dir)
    if saved_model_dir.exists():
        shutil.rmtree(str(saved_model_dir))
    saved_model_dir.mkdir()

    with K.get_session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        k_model.set_weights(weights)

        tf.saved_model.simple_save(
            sess,
            str(saved_model_dir.joinpath('1')),
            inputs={'input': k_model.input},
            outputs=dict((output.name, tensor) for output, tensor in zip(onnx_model.graph.output, k_model.outputs))
        )


def savedmodel2tflite(saved_model_dir, tflite_model_path, quantize=False):
    saved_model_dir = str(Path(saved_model_dir).joinpath('1'))
    converter = tf.contrib.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        converter.optimizations = [tf.contrib.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model
