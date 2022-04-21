import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.data import Data
from skimage import io, filters, feature,transform
import matplotlib.pyplot as plt

from IPython.core.pylabtools import figsize

import os
rf=[]
plt.plot(range(10), rf)
plt.show()
# data = Data()

# x,y=data.get_data(data.TEST_DIR)
# print(y)

# print(type(img))
# img=cv2.imread(os.path.join(os.path.join(data.ORIGINAL_DIR, '00000'), '00000_00001.ppm'))
# print(type(img))
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.subplot(2, 2, 1)
# img = io.imread(os.path.join(os.path.join(data.ORIGINAL_DIR, '00000'), '00000_00000.ppm'))
# plt.axis('off')
# plt.imshow(img)
# plt.title('Original Image')
# plt.subplot(2, 2, 2)
# img = io.imread(os.path.join(os.path.join(data.CROP_DIR, '00000'), '00000_00000.ppm'))
# plt.axis('off')
# img = transform.resize(img, (32, 32))
# plt.imshow(img)
# plt.title('Cropped Image')
# plt.subplot(2, 2, 3)
# img = io.imread(os.path.join(os.path.join(data.CROP_DIR, '00000'), '00000_00000.ppm'),as_gray=True)
# img = transform.resize(img, (32, 32))
# plt.axis('off')
# plt.imshow(img,cmap ='gray')
# plt.title('Grayscale Image')
# plt.subplot(2,2,4)
# mm = MinMaxScaler()
# img=mm.fit_transform(img)
# plt.axis('off')
# plt.imshow(img,cmap ='gray')
# plt.title('MinMaxScaler Image')
# # plt.show()
# # plt.rcParams['figure.figsize'] = (6.4, 2.4)
# plt.savefig('preprocess.png', dpi=300)

# hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# plt.imshow(hsv)
# plt.show()

# edges=cv2.Canny(img, 200, 400)

# edges = feature.canny(img, sigma=1)
# plt.imshow(edges)
# plt.show()

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
# closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
# plt.imshow(closed)
# plt.show()

# edges = filters.sobel(img)
# img=transform.resize(img,(32,32))
# edges=feature.canny(img,sigma=1)
# plt.subplot(121)
# plt.imshow(img)
# plt.subplot(122)
# plt.imshow(edges)
# plt.show()


