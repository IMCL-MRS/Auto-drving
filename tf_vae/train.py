"""
Train a VAE model using saved images in a folder
"""
import argparse
import os,sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from stable_baselines.common import set_global_seeds
from tqdm import tqdm
from config import ROI, INPUT_DIM

from controller import VAEController
from data_loader import DataLoader
from model import ConvVAE


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='Path to a folder containing images for training', type=str, default='train/')
parser.add_argument('--z-size', help='Latent space', type=int, default=128)
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('--n-samples', help='Max number of samples', type=int, default=-1)
parser.add_argument('--batch-size', help='Batch size', type=int, default=64)
parser.add_argument('--learning-rate', help='Learning rate', type=float, default=1e-4)
parser.add_argument('--kl-tolerance', help='KL tolerance (to cap KL loss)', type=float, default=0.5)
parser.add_argument('--beta', help='Weight for kl loss', type=float, default=1.0)
parser.add_argument('--n-epochs', help='Number of epochs', type=int, default=50)
parser.add_argument('--verbose', help='Verbosity', type=int, default=1)
args = parser.parse_args()

set_global_seeds(args.seed)

if not args.folder.endswith('/'):
    args.folder += '/'

vae = ConvVAE(z_size=args.z_size,
              batch_size=args.batch_size,
              learning_rate=args.learning_rate,
              kl_tolerance=args.kl_tolerance,
              beta=args.beta,
              is_training=True,
              reuse=False)

images = [im for im in os.listdir(args.folder) if im.endswith('.png')]
images = np.array(images)
n_samples = len(images)

if args.n_samples > 0:
    n_samples = min(n_samples, args.n_samples)

print("{} images".format(n_samples))

# indices for all time steps where the episode continues
indices = np.arange(n_samples, dtype='int64')
np.random.shuffle(indices)

# split indices into minibatches. minibatchlist is a list of lists; each
# list is the id of the observation preserved through the training
minibatchlist = [np.array(sorted(indices[start_idx:start_idx + args.batch_size]))
                 for start_idx in range(0, len(indices) - args.batch_size + 1, args.batch_size)]

data_loader = DataLoader(minibatchlist, images, n_workers=2, folder=args.folder)

vae_controller = VAEController(z_size=args.z_size,input_dimension=INPUT_DIM)
vae_controller.vae = vae

for epoch in range(args.n_epochs):
    pbar = tqdm(total=len(minibatchlist))
    for obs in data_loader:
        feed = {vae.input_tensor: obs}
        (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
            vae.loss,
            vae.r_loss,
            vae.kl_loss,
            vae.global_step,
            vae.train_op
        ], feed)
        pbar.update(1)
    pbar.close()
    print("Epoch {:3}/{}".format(epoch + 1, args.n_epochs))
    print("VAE: optimization step", (train_step + 1), train_loss, r_loss, kl_loss)
    if epoch % 1 == 0:
        vae_controller.vae.save_checkpoint("vae_"+str(epoch))

        # Update params
        vae_controller.set_target_params()
        # vae_controller.target_vae.save_checkpoint("target_vae_"+str(epoch))
        # Load test image
        if args.verbose >= 1:
            img_data = []
            img_rec = []
            for i in range(16):
                image_idx = np.random.randint(n_samples)
                image_path = args.folder + images[image_idx]
                image = cv2.imread(image_path,0)
                image = cv2.resize(image, (200, 120), interpolation=cv2.INTER_AREA) # width(x), height(y)
                im = image.reshape(INPUT_DIM)
                encoded = vae_controller.encode(im)
                reconstructed_image = vae_controller.decode(encoded)[0]
                # print(image.shape, reconstructed_image.shape)
                out_data = np.hstack((image, reconstructed_image.reshape(120,200)))
                cv2.imwrite(str(i)+".png", out_data)
                # img_data.append(im.reshape(120,200))
                # img_rec.append(reconstructed_image)

                # Plot reconstruction
                # cv2.imshow("Original", image)
                # cv2.imshow("Reconstruction", reconstructed_image)
                # cv2.waitKey(1)

    save_path = "logs/vae-{}".format(args.z_size)
    os.makedirs(save_path, exist_ok=True)
    print("Saving to {}".format(save_path))
    vae_controller.set_target_params()
    vae_controller.target_vae.save_checkpoint("target_vae_" + str(epoch))
    vae_controller.save(save_path)
