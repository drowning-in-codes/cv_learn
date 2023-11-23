import cv2
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
img = cv2.imread("../imgs/00000.png",cv2.IMREAD_GRAYSCALE)
# cv2.imshow("img",img)
# smaller_img = cv2.resize(img,(200,200),interpolation=cv2.INTER_LINEAR)
# cv2.imshow("smaller_img",smaller_img)
rows,cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("dst",dst)


M = np.float32([[1,0,-100],[0,1,-100]])
dst = cv2.warpAffine(img,M,(cols,rows))
plt.imshow(dst)
cv2.imshow("dst",dst)


plt.show()
cv2.waitKey()