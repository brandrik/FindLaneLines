
#**Finding Lane Lines on the Road**

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Pipeline:

The pipeline consists of 7 steps:

1. Conversion to grayscale image
2. Masking (Reducing processed image to region of interest)
3. Removing noise using gaussian blur
4. Canny edge detection
5. Hough line detection
6. Process lines ( removing non lane lines )
7. Overlay the line image onto the original road image
TEST


In order to draw a single line on the left and right lanes, I processed the resulting lines from the Hough line detection in step 5 within in the function *process_lines* before using the *draw_lines* function. The latter one wasn’t changed in functionality.  

####To step 6: Process lines
Within *process_lines* lines with slopes smaller than 0.4 are discarded to avoid lane detection errors caused by horizontal lines.
Then the lines are classified in being part of the left and right lane, respectively.
The average line is computed for each group. There resulting left and right line is then extrapolated. Extrapolation is carried out after averaging, to keep the impact of longer lines on the resulting extrapolated line higher than of shorter lines.
This should reduce the distortion of short lines not marking a lane on the averaged and extrapolated line, which is assumed to be a lane.

#### Images of pipeline
If you'd like to include images to show how the pipeline works, here is how to include an image:

[//]: # (Image References)
[image1]: ./test_images/solidWhiteCurve.jpg "Input"
[image1]: ./test_images/solidWhiteCurve.jpg "Input"
[image1]: ./test_images/solidWhiteCurve.jpg "Input"
[image1]: ./test_images/solidWhiteCurve.jpg "Input"
[image1]: ./test_images/solidWhiteCurve.jpg "Input"
[image1]: ./test_images/solidWhiteCurve.jpg "Input"
[image1]: ./test_images/solidWhiteCurve.jpg "Input"


![Input][image1]

###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ...

Another shortcoming could be ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...

Possible improvements:


An Improved Linear-Parabolic Model for Lane Following
http://ieeexplore.ieee.org/abstract/document/1599093/

real-time road detection application based on neural networks, that might work without lanes
http://ieeexplore.ieee.org/abstract/document/1398891/?reload=true
