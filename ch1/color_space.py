import cv2
import numpy as np
import matplotlib.pylab as plt


img = cv2.imread("../imgs/00000.png")

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.imshow(hsv_img)
plt.show()