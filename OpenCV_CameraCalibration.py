#############################   FIND CHESSBOARD CORNERS   #############################  
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

chessboardSize=(9,6) # Specified chessboard size with number of corners both with width and height.
frameSize= (1280, 720)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
objp = np.zeros((chessboardSize[0] * chessboardSize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)
objpoints = [] 
imgpoints = []
images = glob.glob('chessboard/*.jpg') 

for fname in images:
    # print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(20)
cv2.destroyAllWindows()
print("Object Points: ", objpoints)
print("Image Points: ", imgpoints)

#############################   Calibration   #############################  

ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)                # points that it needs to be specified. 

rvec_average=np.average(rvecs)
tvecs_average= np.average(tvecs)

print("Camera Calibrated: ", ret)
print("\nCamera Matrix: \n",cameraMatrix)
print("\nDistortion Coefficient: \n", dist)
print("\nRotation Vectors: \n", rvec_average)
print("\nTranslation Vectors: \n", tvecs_average)

#############################   Undistorting of the image  ############################# 

img=cv2.imread("road/road.jpg")
h, w= img.shape[:2] 
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h)) 
#Undistort:
dst= cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
#crop the image:
x, y, w, h = roi
dst=dst[y:y+h, x:x+w]
# cv2.imshow('road/Undistort.jpg', dst)
cv2.imwrite('road/Undistort.jpg', dst)
# #Undistort with remapping:
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5) 
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
#crop the image:
x, y, w, h=roi
dst=dst[y:y+h, x:x+w]
# cv2.imshow('road/Remap.jpg', dst)
cv2.imwrite('road/UndistortRemap.jpg', dst)

#Error reprojection 
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ =cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("\ntotal error: {}" .format(mean_error/len(objpoints)))
print("\n\n\n")

# def undistort(img, cameraMatrix, dist):
#     undistort = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
#     return undistort

cap = cv2.VideoCapture("video/motorway_edge.mp4")
# cap = cv2.VideoCapture("video/motorway_cut.mp4")
# cap = cv2.VideoCapture("video/motorway_long.mp4")

while (cap.isOpened()):
    ret, frame = cap.read()
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5) #map for x and y directions for undistort function with remapping.
    dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h=roi
    dst=dst[y:y+h, x:x-200]
    cv2.imshow("undistort video capture", dst)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows