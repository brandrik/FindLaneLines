
#Finding Lane Lines on the Road

The goals / steps of this project are the following:
* Creating a pipeline that finds lane lines on the road
* Reflect on the pipeline a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Pipeline:

The pipeline consists of 7 steps:

1. Conversion to grayscale image
2. Masking (Reducing processed image to region of interest)
3. Removing noise using gaussian blur
4. Canny edge detection
5. Overpaint edges introduced by masking
6. Hough line detection
7. Process lines ( removing non lane lines )
8. Overlay the line image onto the original road image
TEST


In order to draw a single line on the left and right lanes, I processed the resulting lines from the Hough line detection in step 5 within in the function *process_lines* before using the *draw_lines* function. The latter one wasn’t changed in functionality.

#### To step 6: Process lines
* in *process_lines* lines with slopes smaller than 0.4 are discarded to avoid lane detection errors caused by horizontal lines, i.e removing horizontal lines
* grouping remaining lines into left and right lane
* averaging each group yielding an averaged left and right line
* extrapolation of the left and the right line, which is supposed to be marking the left and right lane

Extrapolation is carried out after averaging, to keep the impact of longer lines on the resulting extrapolated line higher than of shorter lines.
This should reduce the distortion of short lines not marking a lane on the averaged and extrapolated line, which is assumed to be a lane.

#### Images of pipeline
If you'd like to include images to show how the pipeline works, here is how to include an image:

[//]: # (Image References)

[0_input]: ./test_images_output/pipeline/0_solidWhiteCurve.jpg "Input"
[1_gray]: ./test_images_output/pipeline/1_gray.jpg "Input"
[2_masked]: ./test_images_output/pipeline/2_masked.jpg "Input"
[3_blurred]: ./test_images_output/pipeline/3_blurred.jpg "Input"
[4_edges]: ./test_images_output/pipeline/4_edges.jpg "Input"
[5_edges_filtered]: ./test_images_output/pipeline/5_edges_filtered.jpg "Input"
[6_line_img]: ./test_images_output/pipeline/6_line_img.jpg "Input"
[7_output]:./test_images_output/pipeline/7_output.jpg "Input"

![Input][0_input]

![Gray][1_gray]

![Masked - region of interest][2_masked]

![Gaussian Blurred][3_blurred]

![Canny detected edges][4_edges]

![Edges - removed edges caused by masking][5_edges_filtered]

![Hough transformation leading to lines detected][6_line_img]

![Weighted - overlay input with lane marking][7_output]





### 2. Shortcomings

The pipeline might not work well enough, if

* if the road is tightly curved
* the car is not centered between the lanes, e.g. when overtaking or exiting the highway
* high contrasts (high gradient color changes, like shadows from trees, or road color changes e.g. concrete to tar) cause edges and therwith lines detected, which are not part of the lanes. This can be seen in the output of the 'challenge.mp4' video.
* when it is dark, or generally different lighting of the road




### 3. Suggest possible improvements to your pipeline

Possible improvements:


* Low pass filtering over frames of the video: This would remove the danger of not having any lead, if marking is not provided or not detected correctly.
* Run hough line detection on parts of the image: divide the image into close (5m ahead of the car), near (5 to 20m)  and far away (20m +). Each part will have its own masking and parameters for improving detection. For example the further away, the more likely to see curved lines if a curve is coming up. Thus filtering of non-lane lines could be adjusted differently
* Using a transformation for curves (Hough lines for circle?)
* Using a total different approach: Adaptive real-time road detection with neural networks, e.g. http://ieeexplore.ieee.org/abstract/document/1398891/?reload=true


* or with a model:
An Improved Linear-Parabolic Model for Lane Following
http://ieeexplore.ieee.org/abstract/document/1599093/

Further possible improvements:


Add Equalizing of the gray-scale image as step 2b. After Removing noise using the gaussian blur a threshold via (cv2.adaptiveThreshold) might make the lane detection more robust against brightness changes (daylight variations, shadows, etc.).

The resulting pipeline would be as follows:
1. Using different color space HSV (Hue, Saturation, Value)
2. Conversion to grayscale image
3. Masking (Reducing processed image to region of interest)
4. Equalized the grayscaled image (e.g. cv2.equalizeHist)
5. Removing noise using gaussian blur
6. Threshold (e.g. cv2.adaptiveThreshold)
7. Skeletonize (skimage.morphology.skeletonize)
8. Overpaint edges introduced by masking
9. Hough line detection
10. Process lines ( removing non lane lines )
11. Overlay the line image onto the original road image
