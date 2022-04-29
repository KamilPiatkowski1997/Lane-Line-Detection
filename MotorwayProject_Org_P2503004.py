#############################   LINE DETECTION SYSTEM   #############################  
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt




#############################   Find chessboard size   #############################  
chessboardSize=(9,6) # Specified chessboard size with number of corners both with width and height.
frameSize= (720, 1280) # Pixel camera frame size in the image for camera calibration.

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # termination criteria default from OpenCV to find subpixels of the corners in the images.

# prepare and compare object points in ideal way for camera calibration, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0).
objp = np.zeros((chessboardSize[0] * chessboardSize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images. 
objpoints = [] # 3d point in real world space. It is needed to project that down to 2d points to find the relation between this two dimentions.
imgpoints = [] # 2d points in image plane. This camera matrix calibration will be used to decrease the camera lens distortion.
images = glob.glob('chessboard/*.jpg') #All images in this folder with the chessboard and .jpg are going to be stored in this variable.
#For loop operation is to run through the stored in the variable images and make operations and store the image and object points in order to undistored the images.
for fname in images:
    # print(fname) #printing all eligible images in particular order.
    img = cv2.imread(fname) #storing variable for all printed images.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None) # Chess board corners can be found in BGR grayscale by this function from OpenCV

    # If found corners true, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) #parameters for finding corner subpixels by OpenCV
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (9,6), corners2, ret) # Draw and display the corners.
        # cv2.imshow('img', img)
        cv2.waitKey(20)
cv2.destroyAllWindows()

print("Object Points: ", objpoints)
print("Image Points: ", imgpoints)

#############################   Calibration Parameters   #############################  
ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)# points that it needs to be specified. 

print("Camera Calibrated: ", ret)
print("\nCamera Matrix: \n",cameraMatrix)
print("\nDistortion Parameters: \n", dist)
print("\nRotation Vectors: \n", rvecs)
print("\nTranslation Vectors: \n", tvecs)

#############################   Error reprojection  #############################  
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ =cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("\ntotal error: {}" .format(mean_error/len(objpoints)))
print("\n\n\n")

#############################   FRAME UNDISTORTION  ############################# 
def undistort(frame, cameraMatrix, dist):
    h, w= frame.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h)) #optimatization on the new camera matrix to recive more accurate resoults
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5) #map for x and y directions for undistort function with remapping.
    dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h=roi
    dst=dst[y:y+h, x:x+w]
    # cv2.imshow('dst', dst)
    return dst
#############################   WARP PERSPECTIVE   #############################  
def warpimage(frame):
    pt1 = np.float32([
        (565,380),    #1
        (750,380),    #2
        (160,530),    #3
        (1140,530)    #4
    ])
    height, width = 360,350
    pt2 = np.float32([[0,0], [width,0], [0, height], [width,height]])
    mtx = cv2.getPerspectiveTransform(pt1,pt2)
    wraped=cv2.warpPerspective(frame, mtx, (width, height))
    cv2.imshow('wraped', wraped)
    return wraped  
#############################   VIEWBOX   ############################# 
def frame_polyfit(frame):
    frame_copy = np.copy(frame)
    polifit= np.float32([(720,380),(585,380),(270,530),(1000,530)])
    frame_poly=cv2.polylines(frame_copy, [np.int32(polifit)], True, (0, 0, 200), 2)
    # cv2.imshow("Frame Polylines", frame_poly)
    return frame_poly
    
#############################   TRESHOLD FOR THE BIRD EYE VIEW   #############################  
def treshold_parameters(wraped):
    R = wraped[:,:,0]
    R_max, R_mean = np.max(R), np.mean(R)
    R_low_white = min(max(150, int(R_max * 0.55), int(R_mean * 1.95)),230)
    R_binary = cv2.inRange(R, R_low_white, 255)
    R_res = cv2.bitwise_and(wraped, wraped, mask= R_binary)
    # cv2.imshow("R chanell", R_binary)

    hsv= cv2.cvtColor(wraped, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array([0,0,170])
    hsv_upper = np.array([255, 40, 220])
    hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    hsv_res = cv2.bitwise_and(wraped, wraped, mask= hsv_mask)
    # cv2.imshow("HSV", hsv_res)

    hls = cv2.cvtColor(wraped, cv2.COLOR_RGB2HLS)
    hls_lower = np.array([0,170,0])
    hls_upper = np.array([255, 255, 255])
    hls_mask = cv2.inRange(hls, hls_lower, hls_upper)
    hls_res = cv2.bitwise_and(wraped, wraped, mask= hls_mask)
    # cv2.imshow("HLS", hls_res)

    combined=np.asarray(hls_res + hsv_res +  R_res, dtype=np.uint8)
    # cv2.imshow("Treshold_combined", combined)
    return combined


# #############################   HISTOGRAM WITH SLIDING WINDOW    ############################# 
def slighting_window(combined):
    dtype=np.int32
    out_combined=np.copy(combined)
    hist=np.sum(combined[combined.shape[0]//2:,:], axis=0)
    midpoint=dtype(hist.shape[0]/2)
    leftx_base=np.argmax(hist[:midpoint]) - midpoint
    rightx_base = np.argmax(hist[midpoint:]) 
    
    # print(leftx_base)
    # print(midpoint)
    # print(rightx_base)

    nwindows = 10
    window_height = dtype(combined.shape[0]/nwindows)
    nonzero = combined.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = combined.shape[0] - (window+1)*window_height
        win_y_high = combined.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + 50
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + 50
        # Draw the windows on the visualization image
        cv2.rectangle(out_combined,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 1) 
        cv2.rectangle(out_combined,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 1) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = dtype(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = dtype(np.mean(nonzerox[good_right_inds]))
    

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    #   Generate x and y values for plotting
    ploty = np.linspace(0, combined.shape[0]-1, combined.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # print(left_fitx)
    # print(right_fitx)

    # Generate black image and colour lane lines
    out_combined[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
    out_combined[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]

    # Draw polyline on image
    right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
    cv2.polylines(out_combined, [right], False, (255,255,1), thickness=5)
    cv2.polylines(out_combined, [left], False, (255,255,1), thickness=5)

    # Drawing the pathway between detected lanes in sliding windows
    final_image=np.copy(combined)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(final_image, np.int_([pts]), (0,255, 0))
    cv2.polylines(final_image, np.int32([pts_left]), False, color=(255,255,1), thickness=25)
    cv2.polylines(final_image, np.int32([pts_right]), False, color=(255,255,1), thickness=25)
    # cv2.imshow("Pathway", final_image)

    # Unwarping the perspective and applying the drawing of pathway on the video capture
    pt1 = np.float32([
        (555,450),    
        (1500,450), 
        (-2800,1800),  
        (9500,1000)
    ])
    height, width = 720, 1280
    pt2 = np.float32([[0,0], [width,0], [0, height], [width,height]])
    mtx = cv2.getPerspectiveTransform(pt2,pt1)
    unwraped=cv2.warpPerspective(final_image, mtx, (width, height))
    # cv2.imshow("Pathway1", unwraped)
    result=cv2.addWeighted(frame,1,unwraped,0.5,0)
    cv2.imshow("result", result)
    # plt.plot(out_combined)
    # plt.plot(hist)
    # plt.show()
  
    return out_combined

#############################   CODE RELEASE    ############################# 
cap = cv2.VideoCapture("video/motorway_edge.mp4")
# cap = cv2.VideoCapture("video/motorway_cut.mp4")
# cap = cv2.VideoCapture("video/motorway_long.mp4")

while (cap.isOpened()):
    ret, frame = cap.read()
    undistort_video = undistort(frame, cameraMatrix, dist)
    polifit_birdeye_section= frame_polyfit(undistort_video)
    bird_eye= warpimage(undistort_video)
    treshold_birdeye = treshold_parameters(bird_eye)
    line_detection = slighting_window(treshold_birdeye)
    # unwarp=unwarpimage(frame)

    # cv2.imshow("unwarp", unwarp)
    # cv2.imshow("Video", polifit_birdeye_section)
    cv2.imshow("Sliding windows", line_detection)
    cv2.imshow("Treshold", treshold_birdeye)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows