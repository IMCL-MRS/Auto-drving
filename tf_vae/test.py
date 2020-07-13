"""
Test a trained vae
"""
import argparse
import os,sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from stable_baselines.common import set_global_seeds
from PIL import Image

from controller import VAEController

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='Log folder', type=str, default='test/')
parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='./logs/vae-128.pkl')
parser.add_argument('--n-samples', help='Max number of samples', type=int, default=1)
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
args = parser.parse_args()

set_global_seeds(args.seed)

if not args.folder.endswith('/'):
    args.folder += '/'

vae = VAEController()
vae.load(args.vae_path)
vae.target_vae.save_checkpoint("test_model")

images = [im for im in os.listdir(args.folder) if im.endswith('.png')]
images = np.array(images)
n_samples = len(images)


for i in range(args.n_samples):
    # Load test image
    image_idx = np.random.randint(n_samples)
    image_path = args.folder + images[image_idx]
    image_path = './input.jpg'
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.expand_dims(np.array(image), axis=2)
    # cv2.imwrite("input.jpg", image)
    print(image.shape)
    image = image.reshape(120, 200, 1)
    # print(image[image > 0])
    encoded = vae.encode_from_raw_image(image)
    print(encoded)
    # print(encoded[encoded > 0])
    reconstructed_image = vae.decode(encoded)[0]
    cv2.imwrite("decode.jpg", reconstructed_image)
    # print(reconstructed_image[reconstructed_image>0])

