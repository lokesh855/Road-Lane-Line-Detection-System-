import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import numpy as np  # Importing numpy for numerical operations
import cv2  # Importing OpenCV for computer vision tasks
import os  # Importing os for interacting with the operating system
import matplotlib.image as mpimg  # Importing matplotlib image for image operations
from moviepy.editor import VideoFileClip  # Importing VideoFileClip from moviepy for video processing
import math  # Importing math for mathematical operations

def interested_region(img, vertices):
    mask = np.zeros_like(img)  # Creating a mask with the same dimensions as the image
    if len(img.shape) > 2:  # Checking if the image is colored
        mask_color_ignore = (255,) * img.shape[2]  # Setting the mask color for colored image
    else:
        mask_color_ignore = 255  # Setting the mask color for grayscale image
        
    cv2.fillPoly(mask, vertices, mask_color_ignore)  # Filling the polygon defined by vertices on the mask
    return cv2.bitwise_and(img, mask)  # Applying the mask to the image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)  # Applying Hough Line Transform
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # Creating an empty image to draw lines
    lines_drawn(line_img, lines)  # Drawing the lines on the empty image
    return line_img  # Returning the image with lines

def lines_drawn(img, lines, color=[255, 0, 0], thickness=6):
    global cache  # Using global variable for cache
    global first_frame  # Using global variable to check if it's the first frame
    slope_l, slope_r = [], []  # Lists to store slopes of left and right lanes
    lane_l, lane_r = [], []  # Lists to store coordinates of left and right lanes
    α = 0.2  # Smoothing factor for lane detection

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)  # Calculating the slope of the line
            if slope > 0.4:  # If the slope is positive and significant, it's a right lane
                slope_r.append(slope)
                lane_r.append(line)
            elif slope < -0.4:  # If the slope is negative and significant, it's a left lane
                slope_l.append(slope)
                lane_l.append(line)
        img_height = img.shape[0]  # Getting the height of the image

    if len(lane_l) == 0 or len(lane_r) == 0:  # If no lanes are detected
        print('No lane detected')
        return

    slope_mean_l = np.mean(slope_l, axis=0)  # Calculating the mean slope of left lanes
    slope_mean_r = np.mean(slope_r, axis=0)  # Calculating the mean slope of right lanes
    mean_l = np.mean(np.array(lane_l), axis=0)  # Calculating the mean coordinates of left lanes
    mean_r = np.mean(np.array(lane_r), axis=0)  # Calculating the mean coordinates of right lanes

    if slope_mean_r == 0 or slope_mean_l == 0:  # Checking for division by zero
        print('Dividing by zero')
        return

    y1 = img_height  # y-coordinate of the bottom of the image
    y2 = int(img_height * 0.6)  # y-coordinate of the point where the lines meet

    x1_l = int((y1 - mean_l[0][1] + (slope_mean_l * mean_l[0][0])) / slope_mean_l)  # Calculating the x-coordinate of the bottom of the left lane
    x2_l = int((y2 - mean_l[0][1] + (slope_mean_l * mean_l[0][0])) / slope_mean_l)  # Calculating the x-coordinate of the top of the left lane
    x1_r = int((y1 - mean_r[0][1] + (slope_mean_r * mean_r[0][0])) / slope_mean_r)  # Calculating the x-coordinate of the bottom of the right lane
    x2_r = int((y2 - mean_r[0][1] + (slope_mean_r * mean_r[0][0])) / slope_mean_r)  # Calculating the x-coordinate of the top of the right lane

    if x1_l > x1_r:  # If the lanes intersect
        x1_l = int((x1_l + x1_r) / 2)
        x1_r = x1_l
        y1_l = int((slope_mean_l * x1_l) + mean_l[0][1] - (slope_mean_l * mean_l[0][0]))
        y1_r = int((slope_mean_r * x1_r) + mean_r[0][1] - (slope_mean_r * mean_r[0][0]))
        y2_l = int((slope_mean_l * x2_l) + mean_l[0][1] - (slope_mean_l * mean_l[0][0]))
        y2_r = int((slope_mean_r * x2_r) + mean_r[0][1] - (slope_mean_r * mean_r[0][0]))
    else:  # If the lanes don't intersect
        y1_l = y1
        y2_l = y2
        y1_r = y1
        y2_r = y2

    present_frame = np.array([x1_l, y1_l, x2_l, y2_l, x1_r, y1_r, x2_r, y2_r], dtype="float32")  # Creating the current frame

    if first_frame == 1:  # If it's the first frame
        next_frame = present_frame        
        first_frame = 0        
    else:  # If it's not the first frame, apply smoothing
        prev_frame = cache
        next_frame = (1 - α) * prev_frame + α * present_frame

    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), color, thickness)  # Drawing the left lane
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), color, thickness)  # Drawing the right lane

    cache = next_frame  # Updating the cache

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)  # Combining the images with weights

def process_image(image):
    global first_frame
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converting the image to grayscale
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # Converting the image to HSV color space
    lower_yellow = np.array([20, 100, 100], dtype="uint8")  # Defining the lower bound for yellow color
    upper_yellow = np.array([30, 255, 255], dtype="uint8")  # Defining the upper bound for yellow color
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)  # Creating a mask for yellow color
    mask_white = cv2.inRange(gray_image, 200, 255)  # Creating a mask for white color
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)  # Combining the yellow and white masks
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)  # Applying the combined mask to the grayscale image
    gauss_gray = cv2.GaussianBlur(mask_yw_image, (5, 5), 0)  # Applying Gaussian blur to the masked image
    canny_edges = cv2.Canny(gauss_gray, 50, 150)  # Performing Canny edge detection
    imshape = image.shape  # Getting the shape of the image
    lower_left = [imshape[1] / 9, imshape[0]]  # Defining the lower left vertex of the region of interest
    lower_right = [imshape[1] - imshape[1] / 9, imshape[0]]  # Defining the lower right vertex of the region of interest
    top_left = [imshape[1] / 2 - imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]  # Defining the top left vertex of the region of interest
    top_right = [imshape[1] / 2 + imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]  # Defining the top right vertex of the region of interest
    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]  # Creating an array of vertices
    roi_image = interested_region(canny_edges, vertices)  # Getting the region of interest from the edge-detected image
    theta = np.pi / 180  # Setting the theta parameter for Hough Line Transform
    line_image = hough_lines(roi_image, 2, theta, 15, 40, 20)  # Performing Hough Line Transform
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)  # Combining the line image with the original image
    return result  # Returning the final image

first_frame = 1  # Initializing the first frame variable
cache = None  # Initializing the cache variable
white_output = 'output_video.mp4'  # Output file name for the processed video
clip1 = VideoFileClip("input_video.mp4")  # Input video file
white_clip = clip1.fl_image(process_image)  # Processing each frame of the video
white_clip.write_videofile(white_output, audio=False)  # Writing the processed video to a file
