import numpy as np
import cv2
from matplotlib import pyplot as plt
import time


imgname1 = r'dormitory1.jpg'
imgname2 = r'dormitory2.jpg'

sift = cv2.SIFT_create()

# FLANN 参数设计
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)


start1 = time.perf_counter()  # 返回系统运行时间

img1 = cv2.imread(imgname1)
img1 = cv2.resize(img1,dsize=(700,700))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
kp1, des1 = sift.detectAndCompute(img1,None)  #des是描述子

img2 = cv2.imread(imgname2) 
img2 = cv2.resize(img2,dsize=(700,700))
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  #灰度处理图像
kp2, des2 = sift.detectAndCompute(img2,None)  #des是描述子

hmerge = np.hstack((gray1, gray2)) #水平拼接
end1 = time.perf_counter()

cv2.imshow("gray", hmerge) #拼接显示为gray
cv2.waitKey(0)
start2 = time.perf_counter()  # 返回系统运行时间

img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255))  #画出特征点，并显示为红色圆圈
img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255))  #画出特征点，并显示为红色圆圈

hmerge = np.hstack((img3, img4)) #水平拼接

end2 = time.perf_counter()

cv2.imshow("SIFT-FLANN-point", hmerge) #拼接显示为gray
cv2.waitKey(0)


start3 = time.perf_counter()  # 返回系统运行时间

matches = flann.knnMatch(des1,des2,k=2)
# matchesMask = [[0,0] for i in range(len(matches))]

good = []
for m,n in matches:
    if m.distance < 0.8*n.distance:
        good.append([m])

# img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

end3 = time.perf_counter()
print('用时：{:.4f}s'.format(end1-start1+end2-start2+end3-start3))
print('用时：{:.4f}s'.format(end3-start3))

cv2.imshow("SIFT-FLANN", img5)
 
cv2.waitKey(0)
cv2.destroyAllWindows()