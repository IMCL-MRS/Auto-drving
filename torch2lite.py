import logging
from pathlib import Path
import numpy as np
import torch
from model import VAE, Learner
from torch.autograd import Variable
from pytorch2keras import pytorch_to_keras
import sys, os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from pytorch2keras import pytorch_to_keras
from converters import pytorch2savedmodel, savedmodel2tflite
from image import load_and_preprocess_image
from tflite import get_tflite_outputs

logger = logging.getLogger()
logger.setLevel(logging.INFO)

device = 'cuda'


class PytorchToKeras(object):
    def __init__(self, pModel, kModel):
        super(PytorchToKeras, self)
        self.__source_layers = []
        self.__target_layers = []
        self.pModel = pModel
        self.kModel = kModel
        tf.keras.backend.set_learning_phase(0)

    def __retrieve_k_layers(self):
        for i, layer in enumerate(self.kModel.layers):
            if len(layer.weights) > 0:
                self.__target_layers.append(i)

    def __retrieve_p_layers(self, input_size):

        input = torch.randn(input_size)
        input = Variable(input.unsqueeze(0))
        hooks = []

        def add_hooks(module):

            def hook(module, input, output):
                if hasattr(module, "weight"):
                    # print(module)
                    self.__source_layers.append(module)

            if not isinstance(module, nn.ModuleList) and not isinstance(module, nn.Sequential) and module != self.pModel:
                hooks.append(module.register_forward_hook(hook))

        self.pModel.apply(add_hooks)

        self.pModel(input)
        for hook in hooks:
            hook.remove()

def main():
    logger.info('Create data directory in which models dumped.\n')
    data_dir = Path.cwd().joinpath('data')
    data_dir.mkdir(exist_ok=True)

    logger.info('\nInitialize vae and load pre-trained weights\n')
    vae = VAE().to(device)
    # vae = Learner().cuda()
    state_dict = torch.load('vae.pkl', map_location='cpu')
    vae.load_state_dict(state_dict)

    input_np = np.random.uniform(0, 1, (1, 1, 120, 200))
    input_var = Variable(torch.FloatTensor(input_np))
    # we should specify shape of the input tensor
    k_model = pytorch_to_keras(vae, input_var, verbose=True)


