## Autonomous Lane Tracking
### Edge device(Coral) steup
* Plugin the coral to Rasbery Pi usb port.
* Power on Rasbery Pi and connect it to wifi router.
* Connect Rasbery Pi with ssh command
* Install tflite_runtime on Rasbery PI: pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_armv7l.whl
* Install blas: sudo apt-get install libatlas-base-dev
* Run the tflite examples: https://coral.ai/docs/accelerator/get-started/#next-steps
