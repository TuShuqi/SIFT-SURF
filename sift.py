import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt

# 1) 以灰度图的形式读入图片
img_1 = cv.imread(r'yifulou3.jpg', cv.IMREAD_GRAYSCALE)
img_2 = cv.imread(r'yifulou2.jpg', cv.IMREAD_GRAYSCALE)

# 2) SIFT特征计算
sift = cv.SIFT_create()

img_kp1, img_des1 = sift.detectAndCompute(img_1, None)
img_kp2, img_des2 = sift.detectAndCompute(img_2, None)

# 3) Flann 特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(img_des1, img_des2, k=2)
goodMatch = []
for m, n in matches:
    # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
    if m.distance < 0.50*n.distance:
        goodMatch.append(m)
# 增加一个维度
goodMatch = np.expand_dims(goodMatch, 1)
print(goodMatch[:20])

img_out = cv.drawMatchesKnn(img_1, img_kp1, img_2, img_kp2, goodMatch[:15], None, flags=2)

cv.imshow('image', img_out)           # 展示图片

cv.waitKey(0)                         # 等待按键按下
cv.destroyAllWindows()                # 清除所有窗口
