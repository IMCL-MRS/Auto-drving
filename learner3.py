import tensorflow as tf
from controller import VAEController
from model import ConvVAE
import glob as gb
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
import random
import numpy as np
import os,sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from config import ROI, INPUT_DIM
import matplotlib.pyplot as plt

img_path = gb.glob("dataset/7_9/*.png")
vae = VAEController()
vae.load("./vae128.pkl")
batch_size = 256

# def load_data():
#     inputs, labels = load_data()
#     return inputs, labels
#
# class learner(object):
#     def __init__(self, batch_size=100, learning_rate=0.0001,is_training=True, reuse=False):
#         self.inputs, self.labels = load_data()
#         pass
#
#     def build_net(self):
#         x = self.z_code = vae_encoder(self.inputs)
#         x = tf.layers.dense(inputs=x, units=32, activation=tf.nn.relu)
#         x = tf.layers.dense(inputs=x, units=32, activation=tf.nn.relu)
#         out = tf.layers.dense(inputs=x, units=2)
#         return out

def build_model():
  model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[1,128]),
    layers.Dense(32, activation='relu'),
    layers.Dense(2)
  ])

  optimizer = tf.keras.optimizers.Adam(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

def getLatentCode(path):
    image = cv2.imread(path, 0)
    image = cv2.resize(image, (200, 120), interpolation=cv2.INTER_AREA)  # width(x), height(y)
    im = image.reshape(INPUT_DIM)
    z=vae.encode_from_raw_image(im)
    return z

def getSpeed(path):
    path = path.replace('dataset/7_9/', '')
    path = path.replace('.png', '')
    path_split = path.split('_')
    x = float(path_split[0])
    z = float(path_split[1])
    return np.array([x,z])

def plot_reg(y_pred, y_true, epoch):
    #y_pred = y_pred.data.cpu().numpy()
    #y_true = y_true.data.cpu().numpy()
    y_pred = np.squeeze(y_pred)
    y_true = np.squeeze(y_true)
    fig = plt.figure()
    sub1 = plt.subplot(1,2,1)
    plt.scatter(y_true[:,0].reshape(1,-1), y_pred[:,0].reshape(1,-1))
    plt.plot(y_true[:, 0].reshape(1, -1), y_true[:, 0].reshape(1, -1), color='red', linewidth=3)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")

    sub2 = plt.subplot(1, 2, 2)
    plt.scatter(y_true[:, 1].reshape(1, -1), y_pred[:, 1].reshape(1, -1))
    plt.plot(y_true[:, 1].reshape(1, -1), y_true[:, 1].reshape(1, -1), color='red', linewidth=3)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")

    plt.subplots_adjust(wspace=0.5)
    if not os.path.exists("./results/7_11/figs/"):
        os.makedirs("./results/7_11/figs/")
    fig.savefig("./results/7_11/figs/" + str(epoch) + ".png")
    plt.close(fig)


def sampleData():
    sample = random.sample(img_path, batch_size)
    xList = []
    yList = []
    for path in sample:
        x = getLatentCode(path)
        y = getSpeed(path)
        xList.append(x)
        yList.append(y)
    y_label=np.array(yList)
    y_label=y_label[:,np.newaxis,:]
    return np.array(xList),y_label


if __name__ == '__main__':
    epoch=3000
    model = build_model()
    #model.load_weights('./checkpoints_7_11/my_checkpoint')

    for i in range(epoch):
        x,y_label=sampleData()
        loss=model.train_on_batch(x,y_label)
        y_pred=model.predict(x)
        print('i:',i,'  ', loss)
        if i % 2==0:
            plot_reg(y_pred, y_label, i)
    model.save_weights('./checkpoints_7_11/my_checkpoint')
    # x,y_label=sampleData()
    # model.evaluate(x,y_label)
    # test_z=getLatentCode('input.jpg')
    # test_z=test_z[np.newaxis,:]
    # speed=model.predict(test_z)
    # print(speed)
    # evaluate
    eval_epoch=40
    for i in range(eval_epoch):
        x,y_label=sampleData()
        x_test=x[0]
        x_test=x_test[np.newaxis,:]
        y_pred=model.predict(x_test)
        print('y_label: ', y_label[0], ' y_predicted: ', y_pred)
        #y_pred = model.predict(x)
        # plot_reg(y_pred, y_label, i)
        # print('validation epoch: ', i)


