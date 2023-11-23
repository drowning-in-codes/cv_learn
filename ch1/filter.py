
#importing the required libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('../imgs/test.png')
cv2.imshow("image",image)
kernel = np.ones((5,5),np.float32)/25
#using the averaging kernel for image smoothening
averaging_kernel = np.ones((3,3),np.float32)/9
filtered_image = cv2.filter2D(image,-1,kernel)
cv2.imshow("avg_filtered_image",filtered_image)
#get a one dimensional Gaussian Kernel
gaussian_kernel_x = cv2.getGaussianKernel(5,1)
gaussian_kernel_y = cv2.getGaussianKernel(5,1)
#converting to two dimensional kernel using matrix multiplication
gaussian_kernel = gaussian_kernel_x * gaussian_kernel_y.T
#you can also use cv2.GaussianBLurring(image,(shape of kernel),standard deviation) instead of cv2.filter2D
filtered_image = cv2.filter2D(image,-1,gaussian_kernel)
cv2.imshow("filtered_image",filtered_image)

edge_kernel = -1*np.ones((3,3),np.float32)
edge_kernel[1,1] = 8
filtered_image = cv2.filter2D(image,-1,edge_kernel)
cv2.imshow("edge_filtered_image_1",filtered_image)
edge_kernel = -1*np.ones((3,3),np.float32)
edge_kernel[1,1] = 5
edge_kernel[0,0] = 0
edge_kernel[2,2] = 0
edge_kernel[0,2] = 0
edge_kernel[2,0] = 0
filtered_image = cv2.filter2D(image,-1,edge_kernel)
cv2.imshow("sharpen_filtered_image",filtered_image)
edge_kernel[1,1] = 4
filtered_image = cv2.filter2D(image,-1,edge_kernel)
cv2.imshow("edge_filtered_image",filtered_image)


cv2.waitKey()