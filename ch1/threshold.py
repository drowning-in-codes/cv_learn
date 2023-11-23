import cv2
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
img = cv2.imread("../imgs/00000.png",cv2.IMREAD_GRAYSCALE)
ret, thresh_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh_binary_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh_trunc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh_tozero = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh_tozero_inv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

# DISPLAYING THE DIFFERENT THRESHOLDING STYLES
names = ['Oiriginal Image', 'BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
images = img, thresh_binary, thresh_binary_inv, thresh_trunc, thresh_tozero, thresh_tozero_inv

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(names[i])
    plt.xticks([]), plt.yticks([])
plt.show()

ret, thresh_global = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# here 11 is the pixel neighbourhood that is used to calculate the threshold value
thresh_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

thresh_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

names = ['Original Image', 'Global Thresholding', 'Adaptive Mean Threshold', 'Adaptive Gaussian Thresholding']
images = [img, thresh_global, thresh_mean, thresh_gaussian]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(names[i])
    plt.xticks([]), plt.yticks([])

plt.show()