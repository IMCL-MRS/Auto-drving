## Autonomous Lane Tracking
### 1. Demo setup 
1. demo: Learning-based lane tracking: 4 robot keep lane tracking and keep the same distance.
2. step1: collect data from reality and simulator.
3. step2: learning-based algorithm
* algorithm1: image + VAE + fc -> action
* algorithm2: image + DL -> two lanes -> action
 
### 2. Edge device(Coral) steup
* Plugin the coral to Raspberry Pi usb port.
* Power on Raspberry Pi and connect it to wifi router.
* Connect Raspberry Pi with ssh command
* Install tflite_runtime on Raspberry Pi: pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_armv7l.whl
* Install blas: sudo apt-get install libatlas-base-dev
* Run the tflite examples: https://coral.ai/docs/accelerator/get-started/#next-steps

### 3. lightspeeur spr2801s + raspberry pi
* https://dev.gyrfalcontech.ai/get-started-with-the-sdk/
