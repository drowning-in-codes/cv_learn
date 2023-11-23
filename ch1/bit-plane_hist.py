import cv2
import torch
from torch import nn
import matplotlib.pylab as plt
import numpy as np

img  = cv2.imread("../imgs/17.tif",cv2.IMREAD_ANYDEPTH)

c1 = np.mod(img,2)
c2 = np.mod(np.floor(img/2),2)
c3 = np.mod(np.floor(img/4),2)
c4 = np.mod(np.floor(img/8),2)
c5 = np.mod(np.floor(img/16),2)
c6 = np.mod(np.floor(img/32),2)
c7 = np.mod(np.floor(img/64),2)
c8 = np.mod(np.floor(img/128),2)
cc = 2*(2*(2*c8+c7)+c6)
print(cc)

to_plot = [img,c1,c2,c3,c4,c5,c6,c7,c8,cc]
fig,axes = plt.subplots(2,5, subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(hspace=0.05, wspace=0.05)
for ax,i in zip(axes.flat, to_plot):
    ax.imshow(i, cmap='gray')

plt.tight_layout()
plt.show()

cv2.waitKey()

def his(img_gray):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("bins")
    plt.ylabel("pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

if __name__ == '__main__':

    his(img)