import utils
import tflite_runtime.interpreter as tflite
import platform
# import tensorflow as tf
import sys, os, io
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from PIL import Image
import numpy as np
import time

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


# def make_interpreter(model_file):
#     model_file, *device = model_file.split('@')
#     print(model_file)
#     return tflite.Interpreter(
#       model_path=model_file,
#       experimental_delegates=[
#           tflite.load_delegate(EDGETPU_SHARED_LIB,
#                                {'device': device[0]} if device else {})
#       ])

def main():
    print("hello world")
    #image = cv2.imread("0.0722970757_0.223937735123.png", 0)
    #image = cv2.resize(image, (200, 120), interpolation=cv2.INTER_AREA)  # width(x), height(y)
    #with open("test.jpg", "rb") as f:
    #    b = io.BytesIO(f.read())
    # 1.create encoder
    model_name = "models/encode.tflite"
    # encode = make_interpreter(model_name)
    encode = tflite.Interpreter(model_name)
    encode.allocate_tensors()
    input_details = encode.get_input_details()
    output_details = encode.get_output_details()

    # 2. set input
    image = Image.open('input.jpg').convert('L')
    image = np.expand_dims(np.array(image), axis=2).astype(np.float32)
    image = np.expand_dims(np.array(image), axis=0)/255. # [1, 120, 200, 1]
    start = time.perf_counter()

    # 3. inference and get output
    encode.set_tensor(input_details[0]['index'], image)
    encode.invoke()  # inference
    z_code = encode.get_tensor(output_details[0]['index'])
    inference_time = time.perf_counter() - start
    print('%.1fms' % (inference_time * 1000))
    print(z_code)

if __name__ == "__main__":
    main()

