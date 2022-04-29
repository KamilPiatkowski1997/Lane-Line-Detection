import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny_function(image):
    gray= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny_image = cv2.Canny(blur, 50,140)
    return canny_image

def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
         slope, intercept = 0.0001, 0
    y1=620
    y2=int(y1*(7.8/10))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    # print(image.shape)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    # print(left_fit_average, 'left')
    # print(right_fit_average, 'right')
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def lines_function(image, pipeline):
    detecting_line=np.zeros_like(image)
    if pipeline is not None:
        for line in pipeline:
            x1, y1, x2, y2 =line.reshape(4)
            cv2.line(detecting_line, (int(x1), int(y1)), (int(x2), int(y2)), (255,1,255), 10)
    return detecting_line

def function_of_intertest(image):
    polygons=np.array([
    [   (800,450),
        (630,450),    
        (420,600),  
        (1000,600)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image= cv2.bitwise_and(image, mask) 
    return masked_image

# img =cv2.imread("road/village.jpg")
# road_image = np.copy(img )
# canny_output=canny_function(road_image)
# section_of_interest = function_of_intertest(canny_output)
# pipeline= cv2.HoughLinesP(section_of_interest, 2, np.pi/180, 80, np.array([]),minLineLength=30, maxLineGap=5)
# average_pipeline = average_slope_intercept(road_image, pipeline)
# pipeline_image=lines_function(road_image,average_pipeline)
# output_road_image= cv2.addWeighted(road_image, 0.8, pipeline_image, 1, 1)
# cv2.imshow("Output1",output_road_image)
# cv2.waitKey(0)
# plt.imshow(img)
# plt.show() 

video=cv2.VideoCapture("video/countryside_long.mp4")
# video=cv2.VideoCapture("video/countryside_cut.mp4")
# video=cv2.VideoCapture("video/countryside_cut1.mp4")
# video=cv2.VideoCapture("video/countryside_road.mp4")


while (video.isOpened()):
    _, frame= video.read()
    canny_output=canny_function(frame)
    section_of_interest = function_of_intertest(canny_output)
    pipeline= cv2.HoughLinesP(section_of_interest, 4, np.pi/180, 90, np.array([]), minLineLength=60, maxLineGap=15)
    average_pipeline = average_slope_intercept(frame, pipeline)
    pipeline_image=lines_function(frame,average_pipeline)
    output_road_image= cv2.addWeighted(frame, 0.6, pipeline_image, 1,1)
    cv2.imshow("output",output_road_image)
    # cv2.imshow("output",function_of_intertest(canny_output))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()