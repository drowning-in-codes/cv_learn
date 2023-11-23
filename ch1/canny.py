import cv2
import numpy as np

img = cv2.imread("../imgs/00000.png")

edges = cv2.Canny(img, 100, 200)

cv2.imshow("edges", edges)
cv2.waitKey()