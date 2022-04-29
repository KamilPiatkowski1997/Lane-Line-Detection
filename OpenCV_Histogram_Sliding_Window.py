import cv2
import numpy as np
from matplotlib import pyplot as plt


dtype=np.int32
combined= cv2.imread("road/thresholdwarp.jpg")
out_combined=np.copy(combined)
hist=np.sum(combined[combined.shape[0]//2:,:], axis=0)
midpoint=dtype(hist.shape[0]/2)
leftx_base=np.argmax(hist[:midpoint]) - midpoint
rightx_base = np.argmax(hist[midpoint:])

print(leftx_base)
print(midpoint)
print(rightx_base)


nwindows = 8
window_height = dtype(combined.shape[0]/nwindows)
nonzero = combined.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
leftx_current = leftx_base
rightx_current = rightx_base
margin = 200
minpix = 100
left_lane_inds = []
right_lane_inds = []
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = combined.shape[0] - (window+1)*window_height
    win_y_high = combined.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
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

# Generate x and y values for plotting
ploty = np.linspace(0, combined.shape[0]-1, combined.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Generate black image and colour lane lines
out_combined[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
out_combined[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]

# Draw polyline on image
right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
cv2.polylines(out_combined, [right], False, (1,1,255), thickness=5)
cv2.polylines(out_combined, [left], False, (1,1,255), thickness=5)

cv2.imshow("img", combined)
cv2.imshow("out img", out_combined)
plt.plot(hist)
plt.show()

cv2.waitKey("q")
cv2.destroyAllWindows()
