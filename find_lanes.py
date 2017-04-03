# main function

# IMPORT
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from image_processing_functions import grayscale, region_of_interest, gaussian_blur, canny, hough_lines, weighted_img, draw_lines
from math_functions import StraightLine, slope, extrapolate_line, line_interception_y_axis

import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#import image_processing_functions
import pdb
from typing import Sequence



#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', 
# for example, call as plt.imshow(gray, cmap='gray')

# 0 PARAMETERS

## Masking: vertices
IMSHAPE = image.shape # (y,x, num channels ) e.g. (540, 960, 3)  imshape[0] -> y, imshape[1] -> x

CENTER_X = int(IMSHAPE[1] / 2)   # half of x scale in the image
CENTER_Y = int(IMSHAPE[0] / 2)   # half of y scale in the image

TOP_LANE_Y_POS        = int(0.58  * IMSHAPE[0])  # [pixel], top of lane y position
OFFSET_X              = int(0.09 * IMSHAPE[1])   # [pixel], defines offset from the center of top of the
                                                 # lane to the right a. left to define masking polygon
BOTTOM_RIGHT_OFFSET_X = int(0.04 * IMSHAPE[1])   # [pixel], defines offset from the center of top of the
                                                 # lane to the right a. left to define masking polygon

BOTTOM_LEFT_X_POS     = int(0.1 * IMSHAPE[1])    # [pixel], bottom left x position of masking polygon

LINE_THICKNESS        = 10                       # thickness of line to mask edges introduced by masking

MIN_SLOPE_LANE        = 0.4                      # compare abs. value of slopes of lines to filter non-lane lines


## Smoothing
GAUSSIAN_BLUR_KERNEL_SIZE    = 3

## Canny: gradient intensity thresholds
LOW_CANNY_GRAD_INTENS_THR    = 200
HIGHER_CANNY_GRAD_INTENS_THR = 290


MIN_LINE_LEN    = 10    # unitless, lower threshold of gradient intensity
MAX_LINE_GAP    = 10    # unitless, upper threshold of gradient intensity

## Hough Transform
## divide hough space into grid with distance 'steps' rho and angle steps 'theta'
RHO = 2                # [pixel], delta of euclidian distance from origin to the line in [pixel]
THETA = np.pi / 180    # [rad], pi/180 = 1 rad

## HOUGH Threshold: minimum vote it should get for it to be considered as a line.
## Number of votes depend upon number of points on the line.
## So it represents the minimum length of line that should be detected.
HOUGH_ACCUMULATION_THR = 20    # number of votes in accumulation matrix,
                               # this is also the lower bound for minimal line length

## Overlay line image onto original: weights
α = 0.8   # weight for original image
β = 1     # weight for image being overlaid
λ = 0



# 1 FUNCTIONS #### 

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    #pdb.set_trace()
    # PIPELINE
        ## 1) GRAY SCALE
        ## 2) MASKING
        ## 3) 
        ## 4) 
        ## 5) Overlay
    
    # 1 CONVERT TO GRAY SCALE 
    gray = grayscale(image)  # returns one color channel, needs to be set to gray when using imshow()
    
    
    # 2 MASKING - REGION OF INTEREST
    ## reduce the number of pixels to be processed, to lower reduce computational effort for e.g gaussian blur
    ## compute vertices for triangle masking
    ## This time we are defining a four sided polygon to mask
    
    ## (0, ymax), (0,0), (xmax/2-offset, 0) , (xmax/2+offset, ymax)
    
    (X0, Y0) = (BOTTOM_LEFT_X_POS, IMSHAPE[0])                    # left bottom point
    (X1, Y1) = (CENTER_X - OFFSET_X, TOP_LANE_Y_POS)              # left top point
    (X2, Y2) = (CENTER_X + OFFSET_X, TOP_LANE_Y_POS)              # right top point
    (X3, Y3) = (IMSHAPE[1] - BOTTOM_RIGHT_OFFSET_X, IMSHAPE[0])   # right bottom point
    
    vertices = np.array([[(X0, Y0), (X1, Y1), (X2, Y2), (X3, Y3)]], dtype=np.int32)
    masked = region_of_interest(gray, vertices)
    
    
    # 3 REMOVING NOISE WITH GAUSSIAN-BLUR
    ## reduce noise in the gray-scale image to improve later edge detection
    ## This is an extra smoothing applied prior to the one embedded in the canny function
    blurred = gaussian_blur(masked, GAUSSIAN_BLUR_KERNEL_SIZE) # use kernel with size 5
    
    
    # 4 CANNY EDGE DETECTION
    edges = canny(blurred, LOW_CANNY_GRAD_INTENS_THR, HIGHER_CANNY_GRAD_INTENS_THR)  # image w/ edges emphasized

    
    # overpaint edge introduced by prior masking
    cv2.line(edges,(X0,Y0),(X1,Y1),(0),LINE_THICKNESS)
    cv2.line(edges,(X1,Y1),(X2,Y2),(0),LINE_THICKNESS)
    cv2.line(edges,(X2,Y2),(X3,Y3),(0),LINE_THICKNESS)
    

    # 5 HOUGH TRANSFORMATION FOR LINE DETECTION
    lines = hough_lines(edges, RHO, THETA, HOUGH_ACCUMULATION_THR, MIN_LINE_LEN, MAX_LINE_GAP, MIN_SLOPE_LANE, 1, TOP_LANE_Y_POS, IMSHAPE[0])

    # 6 FILTER LANE LINES, DRAW LINES
    
    
    
  
    #simg  = edges
    #import pdb; pdb.set_trace()
    #lines = cv2.HoughLinesP(edges, RHO, THETA, HOUGH_ACCUMULATION_THR,np.array([]) ,MIN_LINE_LEN, MAX_LINE_GAP)
    #line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #print("lines:", lines)
    #print("type ", types(lines[0]))
    
    

    #newlines = [Line(*line[0]) for line in lines]
    #print("newlines ", newlines)
    
    #print("line", lines[0])
    #line = lines[4]
    #print("line new", line[0][1])
    #print("slope", slope(line[0]))
    #draw_lines(line_img, lines)    
    #lines   = line_img
    
    # 6 OVERLAY IMAGES: overlay line image on top of the irginal one.
    weighted = weighted_img(lines, image, α, β, λ)
    
    
    #return edges, masked_edges, weighted
    return weighted


marked = process_image(image)
plt.imshow(marked)


