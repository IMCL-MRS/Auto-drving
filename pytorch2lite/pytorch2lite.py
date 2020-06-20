import logging
from pathlib import Path
import numpy as np
import torch
from model import VAE, Learner
from torch.autograd import Variable
import sys, os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from converters import pytorch2savedmodel, savedmodel2tflite
from image import load_and_preprocess_image
from tflite import get_tflite_outputs
import tensorflow as tf

logger = logging.getLogger()
logger.setLevel(logging.INFO)

device = 'cuda'

def main():
    logger.info('Create data directory in which models dumped.\n')
    data_dir = Path.cwd().joinpath('data')
    data_dir.mkdir(exist_ok=True)
    #
    # logger.info('\nInitialize vae and load pre-trained weights\n')
    # vae = VAE().to(device)
    # # vae = Learner().cuda()
    # state_dict = torch.load('vae.pkl', map_location='cpu')
    # vae.load_state_dict(state_dict)
    # vae.eval()
    #
    # # for m in vae.modules():
    # #     m.training = False
    #
    # # dummy_input1 = torch.randn(1, 1, 120, 200).cuda()
    # input_np = np.random.uniform(0, 1, (1, 1, 120, 200))
    # dummy_input1 = Variable(torch.FloatTensor(input_np)).to(device)
    # # dummy_input = torch.randn(1, 128).cuda() # action
    # input_names = ['input']
    # input_names = ['input1','input2']
    # output_names = ['output']
    #
    # logger.info('\nExport PyTorch model in ONNX format to {onnx_model_path}.\n')
    # torch.onnx.export(vae, dummy_input1, "vae.onnx", verbose=True, input_names=input_names, output_names=output_names)
    #
    # CD_output = CD_graph.get_tensor_by_name('output')
    # x_single = tf.placeholder(tf.float32, [1, 120, 200, 1], name="input1")
    # converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph("vae.pb", INPUT_NODE, OUTPUT_NODE)

    saved_model_dir = str(data_dir.joinpath('save_model'))
    # logger.info('\nConvert ONNX model to Keras and save as saved_model.pb.\n')
    # pytorch2savedmodel("vae.onnx", saved_model_dir)
    #
    logger.info('\nConvert saved_model.pb to TFLite model.\n')
    tflite_model_path = str(data_dir.joinpath('model.tflite'))
    tflite_model = savedmodel2tflite(saved_model_dir, tflite_model_path, quantize=True)

    logger.info('\nConvert saved_model.pb to TFLite quantized model.\n')
    tflite_quantized_model_path = str(data_dir.joinpath('model_quantized.tflite'))
    tflite_quantized_model = savedmodel2tflite(saved_model_dir, tflite_quantized_model_path, quantize=True)



if __name__ == '__main__':
    main()
