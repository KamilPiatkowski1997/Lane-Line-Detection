############################   Generate Thresholded Binary image  ############################# 
# import numpy as np
# import cv2

# img = cv2.imread('road/warp.jpg')
# result = img.copy()
# treshold_image = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
# lower = np.array([0,8,158])
# upper = np.array([29, 31, 255])
# mask = cv2.inRange(treshold_image, lower, upper)
# result = cv2.bitwise_and(result,result, mask=mask)

# cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area < 1:
#         cv2.drawContours(result, [c], -1, (0,0,0), -1)



# cv2.imshow('orginal warp', img)
# cv2.imshow('mask', mask)
# cv2.imwrite('road/mask.jpg', mask)
# cv2.imshow('result', result)
# cv2.waitKey()
# cv2.destroyAllWindows()

#############################   Threshold Window Trackbar    #############################
import cv2
import sys
import numpy as np

def nothing(x):
    pass

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

img = cv2.imread('road/warp.jpg')
output = img
waitTime = 33

while(1):

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img,img, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    cv2.imshow('warp',output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(waitTime) & 0xFF == ord('q'):
        cv2.imwrite('road/thresholdwarp.jpg', mask)
        break
cv2.destroyAllWindows()


#############################   Colour Space Split Function    #############################

# img = cv2.imread('road/warp.jpg')
# cv2.imshow("road",img)

# b,g,r = cv2.split(img)
# b = img[:,:,2]
# g = img[:,:,1]
# r = img[:,:,0]


# cv2.imshow("blue",b)
# cv2.imshow("green",g)
# cv2.imshow("red",r)

# print(b.shape)
# print(g.shape)
# print(r.shape)

# cv2.waitKey() 
# cv2.destroyAllWindows()