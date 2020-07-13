import random
import os
import cv2 as cv
from skimage.transform import rotate, AffineTransform, warp

files = []
name = "./test/0.0722970757_0.223937735123.png"
for i in range(10):
    files.append(name)
if not os.path.exists("./wraps/"):
    os.makedirs("./wraps")

for idx, img_name in enumerate(files):
    base_name = os.path.basename(img_name)
    image = cv.imread(img_name)
    h, w, c = image.shape
    im = cv.resize(image, (200, 120), interpolation=cv.INTER_AREA)
    angle = 0
    if random.choice([True, False]):
        angle = random.randint(-50, 50)
        transform = AffineTransform(translation=(angle, 0))  # (-200,0) are x and y coordinate
        im = warp(im, transform, mode="constant") * 255.
        # im = np.vstack((im, warp_image))
    cv.imwrite("wraps/wrap_" + str(angle) +"_" + base_name, im )


#REF:https://github.com/govinda007/Images/blob/master/augmentation.ipynb