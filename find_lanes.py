# main function

# IMPORT
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import argparse
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from typing import Sequence

#import pdb

from lib.math_functions import StraightLine, slope, extrapolate_line, line_interception_y_axis
from lib.image_processing_functions import grayscale, region_of_interest, gaussian_blur, canny, \
    hough_lines, weighted_img, draw_lines, remove_lines, average_straight_lines, extrapolate_line




# 0 PARAMETERS


## Masking: vertices
# Run determine_params to set image dimension dependent parameters

IMSHAPE = np.array([0,0,0])

TOP_LANE_Y_POS        = int(0)   # [pixel], top of lane y position  0.58

VERTICES = np.array([])

LINE_THICKNESS        = 10                       # thickness of line to mask edges introduced by masking

MIN_SLOPE_LANE        = 0.4                      # compare abs. value of slopes of lines to filter non-lane lines


## Smoothing
GAUSSIAN_BLUR_KERNEL_SIZE    = 3

## Canny: gradient intensity thresholds
LOW_CANNY_GRAD_INTENS_THR    = 220
HIGHER_CANNY_GRAD_INTENS_THR = 290


MIN_LINE_LEN    = 10    # unitless, lower threshold of gradient intensity
MAX_LINE_GAP    = 10    # unitless, upper threshold of gradient intensity

## Hough Transform
## divide hough space into grid with distance 'steps' rho and angle steps 'theta'
RHO = 2               # [pixel], delta of euclidian distance from origin to the line in [pixel]
THETA = np.pi / 360    # [rad], pi/180 = 1 rad

## HOUGH Threshold: minimum vote it should get for it to be considered as a line.
## Number of votes depend upon number of points on the line.
## So it represents the minimum length of line that should be detected.
HOUGH_ACCUMULATION_THR = 10    # number of votes in accumulation matrix,
                               # this is also the lower bound for minimal line length

## Overlay line image onto original: weights
α = 0.8   # weight for original image
β = 1     # weight for image being overlaid
λ = 0


BLACK = [0, 0, 0]


def main():
    """business logic for when running this module as the primary one"""
    parser = argparse.ArgumentParser(description="Mark lane lines on an image of a road.")
    parser.add_argument("src_path", help="Path of source image to have lane lines marked.")
    parser.add_argument("dest_path", help="Path of image with marked lane lines.")
    args = parser.parse_args()
    find_lanes(args.src_path, args.dest_path)
    print("Added lane markings to image:", args.src_path, "and saved the resulting image to:", args.dest_path, "Drive safe!")    


def find_lanes(source_file_path: str, dest_file_path: str) -> None:
    """Mark lane lines in given road image"""
    #reading in an image
    image = mpimg.imread(source_file_path)
    marked = process_image(image)
    save_image(dest_file_path, marked)
    print("Image with marked lanes was written to: ", dest_file_path)


def determine_params(image: np.ndarray):
    """Sets parameters according to image dimensions"""
    
    imshape  = image.shape # (y,x, num channels ) e.g. (540, 960, 3)  imshape[0] -> y, imshape[1] -> x
    
    center_x = int(imshape[1] / 2)   # half of x scale in the image
    center_y = int(imshape[0] / 2)   # half of y scale in the image
    
    top_lane_y_pos        = int(0.6 * imshape[0])   # [pixel], top of lane y position  0.58
    offset_x              = int(0.09 * imshape[1])   # [pixel], defines offset from the center of top of the   0.09
                                # lane to the right a. left to define masking polygon
    bottom_right_offset_x = int(0.04 * imshape[1])   # [pixel], defines offset from the center of top of the
                                # lane to the right a. left to define masking polygon
    
    bottom_left_x_pos     = int(0.1 * imshape[1])    # [pixel], bottom left x position of masking polygon  
    
    
    return (imshape, center_x, center_y, top_lane_y_pos, offset_x, bottom_right_offset_x, bottom_left_x_pos)
    


# 1 FUNCTIONS #### 

def process_image(image: np.ndarray) -> np.ndarray:
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    
    # PIPELINE
        ## 1) GRAY SCALE
        ## 2) MASKING
        ## 3) REMOVING NOISE USING GAUSSIAN BLUR
        ## 4) CANNY EDGE DETECTION
        ## 5) HOUGHT LINE DETECTION (also removing non-lane lines)
        ## 6) Overlay detected lanes on original image
    
    
    IMSHAPE, TOP_LANE_Y_POS, VERTICES = preprocess(image)

    # 1 CONVERT TO GRAY SCALE 
    gray = grayscale(image)  # returns one color channel, needs to be set to gray when using imshow()
    
    # optional thresholding
    ret,thresh=cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    
    # 2 MASKING - REGION OF INTEREST
    ## reduce the number of pixels to be processed, to lower reduce computational effort for e.g gaussian blur
    ## compute vertices for triangle masking
    ## This time we are defining a four sided polygon to mask
    masked = region_of_interest(thresh, VERTICES)
    
    
    ## 3 REMOVING NOISE WITH GAUSSIAN-BLUR
    ### reduce noise in the gray-scale image to improve later edge detection
    ### This is an extra smoothing applied prior to the one embedded in the canny function
    blurred = gaussian_blur(masked, GAUSSIAN_BLUR_KERNEL_SIZE) 
    
    
    # 4 CANNY EDGE DETECTION
    edges = canny(blurred, LOW_CANNY_GRAD_INTENS_THR, HIGHER_CANNY_GRAD_INTENS_THR)  # image w/ edges emphasized

    
    ## overpaint edge introduced by prior masking, so they are not reccoginzed by hough_line method
  
    masking_lines = vertices_to_lines(VERTICES)
    draw_lines(edges, masking_lines, BLACK, LINE_THICKNESS)
    

    # 5 HOUGH TRANSFORMATION FOR LINE DETECTION
        # lines not representing a lane are removed from the result
    lines = hough_lines(edges, RHO, THETA, HOUGH_ACCUMULATION_THR, MIN_LINE_LEN, MAX_LINE_GAP)  # Sequence[StraightLine]
   
    lines = process_lines(lines, MIN_SLOPE_LANE, TOP_LANE_Y_POS, IMSHAPE[0]) # remove non-lane lines 
    line_img = np.zeros((IMSHAPE[0], IMSHAPE[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines) # draw detected lanes into empty image
    
    
    # 6 OVERLAY IMAGES: overlay line image on top of the irginal one.
    weighted = weighted_img(line_img, image, α, β, λ)
    
    
    #return edges, masked_edges, weighted
    return weighted



def compute_vertices(imshape: np.ndarray, center_x, center_y, top_lane_y_pos, offset_x, bootom_right_offset_x, bottom_left_x_pos) -> np.ndarray:

    """ Computes vertices for masking
    Returns: vertices as np.array """

    (X0, Y0) = (bottom_left_x_pos, imshape[0])                    # left bottom point
    (X1, Y1) = (center_x - offset_x, top_lane_y_pos)              # left top point
    (X2, Y2) = (center_x + offset_x, top_lane_y_pos)              # right top point
    (X3, Y3) = (imshape[1] - bootom_right_offset_x, imshape[0])   # right bottom point
    
    vertices = np.array([[(X0, Y0), (X1, Y1), (X2, Y2), (X3, Y3)]], dtype=np.int32)


    return vertices

def vertices_to_lines(vertices: np.ndarray) -> Sequence[StraightLine]:
    """ INPUT:   Coordinate pairs in array: [X0, Y0, X1, Y1], [...], []
        RETURNS: lines: line(X0= , Y0=, X1=, X2= ) """
    
    i = 0
    max_index = vertices.shape[1] - 1
    lines = [[vertices[0][0][0], vertices[0][0][1], vertices[0][max_index][0], vertices[0][max_index][1]]] # append last line
    while i < (vertices.shape[1]-1):     # number of points
        lines = np.append(lines, [[vertices[0][i][0], vertices[0][i][1], vertices[0][i+1][0], vertices[0][i+1][1]]], axis=0)
        i = i + 1
    
    lines = [StraightLine(*line) for line in lines]
    return lines

def process_lines(lines: Sequence[StraightLine], min_slope: int, ymin: int, ymax: int) -> Sequence[StraightLine]:
    """Returns: Sequence of lines fulfilling min slop condition
       trying to remove everything that is not assumed to be a lane line"""

    # 1 filter and separate  ine segments
    # 2 average the position of each of the lines
    # 3 extrapolate to the top and bottom of the lane

    # Note: average first before extrapolating to keep stronger imapct of longer lines on the average lines

    # 1 FILTER LINES (only consider line beeing lane lines)
        # filter by slope: remove horizontal lines, when filtering in left and right lane group

    negative_slope_lines = remove_lines(lines, -1, -min_slope)
    positive_slope_lines = remove_lines(lines, min_slope, 1)

    # 2 AVERAGE the position of the lines
    negative_slope_line = average_straight_lines(negative_slope_lines)
    positive_slope_line = average_straight_lines(positive_slope_lines)

    # 3 EXRTAPOLATE
     
    # only extrapolate if there is a line left in the lists
    extrapolated_neg_slope_line = extrapolate_line(negative_slope_line, ymin, ymax) 

    extrapolated_pos_slope_line = extrapolate_line(positive_slope_line, ymin, ymax)

    lanes = [extrapolated_neg_slope_line] + [extrapolated_pos_slope_line] # combine to one list

    return (list(filter(None.__ne__, lanes)))



def preprocess(image: np.ndarray):
    (imshape, center_x, center_y, top_lane_y_pos, offset_x, bottom_right_offset_x, bottom_left_x_pos) = determine_params(image)
    vertices = compute_vertices(imshape, center_x, center_y, top_lane_y_pos, offset_x, bottom_right_offset_x, bottom_left_x_pos)
    
    return(imshape, top_lane_y_pos, vertices)


def save_image(dest_file_path: str, image: np.ndarray):
    """Returns: True if file was written, false if not"""
    
    dir = os.path.split(dest_file_path)[0]
    if not os.path.exists(dir):
        os.makedirs(dir)    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    saved = cv2.imwrite(dest_file_path, img)
    return saved

# only run the code below, if this script is the entry-point
if __name__ == '__main__':
    main()      
