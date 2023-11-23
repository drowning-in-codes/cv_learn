import cv2
import numpy as np

img = cv2.imread("../imgs/test.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.int8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations = 2)

sure_bg = cv2.dilate(opening,kernel,iterations = 3)

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

ret,sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg,sure_bg)
ret,markers, = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers==-1] = [255,0,0]

# cv2.imshow("img",img)
# cv2.imshow("sure_bg",sure_bg)
# cv2.imshow("sure_fg",sure_fg)
ret,mask = cv2.threshold(sure_bg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
finial = cv2.bitwise_and(img,img,mask=mask)
cv2.imshow("finial",finial)
cv2.imshow("mask",mask)
cv2.imshow("sure_fg",sure_fg)
cv2.imshow("sure_bg",sure_bg)
cv2.imshow("img",img)

cv2.waitKey()