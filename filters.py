import cv2
import numpy as np

# The most basic OpenCV image type is np.uint8. It is a 1-byte unsigned integer, which means it can store values from 0 to 255.
# https://medium.com/@manikantmnnit/understanding-image-quantization-converting-uint8-images-to-float32-for-dl-models-c890c7ee4fc2
# quantization
def quantize(img):
    return np.clip(img, 0, 255).astype(np.uint8)



"""GAMMA FILTER. Increase gamma to darken image.
Gamma must be set above 1 to darken image."""
# https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
def darken(img, gamma=2.0):
    table = np.empty((1,256), np.uint8)
    for i in range(256):
        table[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    return cv2.LUT(img,table)



"""GAUSSIAN BLUR FILTER"""
def gausblur(img, blur_level):
    return cv2.GaussianBlur(img,blur_level,0)



""""GAUSSIAN NOISE FILTER"""
# https://numpy.org/devdocs/reference/generated/numpy.clip.html
def gausnoise(img, std, mean):
    noise = np.random.normal(mean, std, img.shape)
    noise_img = img.astype(np.float32) + noise

    return quantize(noise_img)
