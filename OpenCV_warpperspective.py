#############################   Warping Images  ############################# 
import cv2 
import numpy
import matplotlib.pyplot as plt
# 
# img = cv2.imread("road/Undistort.jpg")
img = cv2.imread("road/UndistortRemap.jpg")
pt1 = numpy.float32([ (600,385),(710,385),(290,530),(960,530)])
height, width = 360,350
pt2 = numpy.float32([[0,0], [width,0], [0, height], [width,height]])
mtx = cv2.getPerspectiveTransform(pt1,pt2)
wraped=cv2.warpPerspective(img, mtx, (width, height))

cv2.imwrite("road/warp.jpg",wraped)
cv2.imshow("Warp",wraped )
cv2.imshow("Original",img )
cv2.waitKey(0)

# plt.imshow(img)
# plt.show() 

cv2.destroyAllWindows