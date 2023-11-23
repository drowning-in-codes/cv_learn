
import numpy as np
import cv2
import matplotlib.pyplot as plt

#reading images in grayscale format
image1 = cv2.imread('../imgs/00000.png',0)
M = cv2.getRotationMatrix2D((image1.shape[1]/2,image1.shape[0]/2),76,1)
image2 = cv2.warpAffine(image1, M, (image1.shape[1], image1.shape[0]))

sift  = cv2.xfeatures2d.SIFT_create()
#finding out the keypoints and their descriptors
keypoints1,descriptors1 = sift.detectAndCompute(image1,None)
keypoints2,descriptors2 = sift.detectAndCompute(image2,None)

#matching the descriptors from both the images
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1,descriptors2,k = 2)


#selecting only the good features
good_matches = []
for m,n in matches:
    if m.distance < 0.25*n.distance:
        good_matches.append([m])
image3 = cv2.drawMatchesKnn(image1,keypoints1,image2,keypoints2,good_matches,None,flags=2)
cv2.imshow("image3",image3)
cv2.waitKey()