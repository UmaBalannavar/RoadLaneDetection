#load and display images 
import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope,intercept=line_parameters #get slope and intercept
    y1=image.shape[0] #height of image
    y2=int(y1*(3/5)) #y2 is 3/5 of the height of image
    x1=int((y1-intercept)/slope) #x1 is the x coordinate of the line
    x2=int((y2-intercept)/slope) #x2 is the x coordinate of the line
    return np.array([x1,y1,x2,y2]) #return the coordinates of the line

def average_slope_intercept(image, lines):
    left_fit = [] #list to store left lines
    right_fit = [] #list to store right lines
    for line in lines: #for each line
        x1,y1,x2,y2=line.reshape(4) #for each point in the line
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        if slope<0:
            left_fit.append((slope,intercept)) #append to left lines
        else: #if slope is positive
            right_fit.append((slope,intercept)) #append to right lines

    left_fit_average=np.average(left_fit,axis=0) #average of left lines
    right_fit_average=np.average(right_fit,axis=0) #average of right lines
    left_line=make_coordinates(image,left_fit_average) #make coordinates for left line
    right_line=make_coordinates(image,right_fit_average) #make coordinates for right line
    return np.array([left_line,right_line]) #return the coordinates of the lines


def canny(lane_image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)  # Use lane_image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image,lines):
    line_image=np.zeros_like(image) #create a blank image to draw lines on
    if lines is not None: #if lines are detected
        for x1,y1,x2,y2 in lines: #for each line
             #reshape the line to get x1,y1,x2,y2
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10) #draw the line on the blank image
    return line_image #return the image with lines        

def region_of_interest(image):
    height=image.shape[0] #height of image
    polygons=np.array([
        [(200, height), (1100, height), (550, 250)]])
    mask=np.zeros_like(image) #create a mask of zeros
    cv2.fillPoly(mask,polygons,255) #fill the mask with white color
    masked_image=cv2.bitwise_and(image,mask) #bitwise and operation to get the region of interest
    return masked_image

def process_image():
    image = cv2.imread("test_image.jpg")
    lane_image = np.copy(image)  # copy of image

    canny_image = canny(lane_image)  # canny edge detection
    cropped_image = region_of_interest(canny_image)  # region of interest
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)  # hough lines
    if lines is not None:
        averaged_lines = average_slope_intercept(lane_image, lines)  # average slope and intercept
        line_image = display_lines(lane_image, averaged_lines)  # display lines
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)  # combine the images
        cv2.imshow('Image Detection', combo_image)  # show the image
        cv2.waitKey(5000)  # Display for 5 seconds (adjust as needed)
        cv2.destroyWindow('Image Detection')  # Close the image window


def process_video():
    cap = cv2.VideoCapture("test2.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break
        canny_image = canny(frame)  # canny edge detection
        cropped_image = region_of_interest(canny_image)  # region of interest
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)  # hough lines
        if lines is not None:
            averaged_lines = average_slope_intercept(frame, lines)  # average slope and intercept
            line_image = display_lines(frame, averaged_lines)  # display lines
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)  # combine the images
            cv2.imshow('Video Detection', combo_image)  # show the image
        key=cv2.waitKey(10) & 0xFF 
        #print(f"key pressed: {key}")
        if key== ord("q"):  # Press 'q' to quit
            break
    cap.release()  # release the video
    cv2.destroyAllWindows()  # destroy all windows


# Run both image and video detection
process_image()
process_video()