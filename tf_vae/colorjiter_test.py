import cv2 as cv
from PIL import Image
import numpy as np
from PIL import ImageEnhance
import random

transform_type_dict = dict(
    brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
    sharpness=ImageEnhance.Sharpness, color=ImageEnhance.Color
)

class ColorJitter(object):
    def __init__(self, transform_dict):
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]

    def __call__(self, img):
        out = img
        rand_num = np.random.uniform(0, 1, len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_num[i] * 2.0 - 1.0) + 1  # r in [1-alpha, 1+alpha)
            print(alpha)
            out = transformer(out).enhance(r)
        return out

img_name = "./test/0.0722970757_0.223937735123.png"
image = cv.imread(img_name)
for i in range(10):
    h, w, c = image.shape
    im = cv.resize(image, (w, h), interpolation=cv.INTER_AREA)
    # 1. color jitter
    if random.choice([True, False]):
        rand_b = random.uniform(0, 0.5)
        rand_ct = random.uniform(0, 0.5)
        rand_s = random.uniform(0, 0.5)
        rand_cl = random.uniform(0, 0.1)
        _transform_dict = {'brightness': rand_b, 'contrast': rand_ct, 'sharpness': rand_s, 'color': rand_cl}
        _color_jitter = ColorJitter(_transform_dict)
        im = Image.fromarray(im)
        im = _color_jitter(im)
        # 做完color jitter之后，再将Image对象转回numpy array
        im = np.array(im)
    # 2. gaussian filter
    im = cv.GaussianBlur(im, (3, 3), 0)

    cv.imwrite(img_name[:-4]+ str(i) + "_cj.png", im)
