## Writeup Template

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

In the following I will illustrate how I achieve these goals.

### Camera Calibration Matrix and Distortion Coefficients from chessboard images.

First, I need to find cross points in the image, I called `image points` which is in 2D space. And I define real chessboard corners in real world as `object points`. Given `object points` pattern, I look for the same pattern in chessboard images and find all the `image points`, then those points are input to function `cv2.calibrateCamera()` to find the camera calibration matrix and also the distortion coefficients. Here is the origin chessboard image:

![alt text][origin_chessboard]

Then I find all corners (image points) as shown by color dots:

![alt text][draw_chessboard]

These points of all chessboard images help me to undistort images. Undistorted chessboard and road image:

![alt text][undistort_chessboard]

![alt text][undistort road]


### Color Thresholding and Gradients Thresholding

Thresholding tricks can extract the lane lines information. Here I depoly the color thresholding and gradients thresholding techniques along with other little tricks.

#### Color Processing

Color gives us a priori knowledge of what lanes look like. Lanes are usually bright yellow and bright white, in other words, those color is high-saturation. It is a good way to extract lanes lines by thresholding images' saturation value. Instead of using RGB color space, we use HLS which has one degree to measure saturation. Here is saturation-thresholded image:

![alt text][color_thresh]

Before going further to gradient thresholding which aims at outlining edges of images, I try to make those edges more prominent with color tricks. In some situation, the road is bright and differences (gradients) between lanes lines and road area becomes indistinguishable. My thought is enlarge this difference by enhancing the lane lines color. As I know where the lane lines are situated from threshold image above, I use it as a mask to select the area to be enhances (The enhancement method is increasing lightness and saturation). Enhanced image is shown below:

![alt text][color_enhance]

#### Gradient Thresholding

Gradient describes the extent of color changes in the image. It is a good metric for detecting edges. Here I use Four gradient thresholding techniques: X-axis gradient thresholding, Y-axis gradient thresholding, Overall gradient thresholding and Direction-gradient thresholding. Thresholded images are displayed in the order: 

![alt text][x_gradient]

![alt text][y_gradient]

![alt text][mag_gradient]

![alt text][dir_gradient]

#### Combination

With thresholded images above, I combine those images to better distinguish lane lines :

![alt text][combo]

### Region Masking

Additionally, the lane area ahead car is where we focus. I use region masking: 

![alt text][region_mask]

The green bouding box outlines the region of interest.

### Lane Area Finding

Given masked image above, pixels of lane lines are approximately selected. Here I use Hough Algorithm to detect lines and then choose pixels of detected lines to approximate the lane lines with linear regression. Fitted lines for lane lines are shown with bold lines:

![alt text][hough_line]

Those lines are drawn so bold that they cover the possible lane lines completely. Then it is used as a mask to filter out noise outside the bold lines effectively.

As I have approximated lane lines, it's apparent where lane area is (within green lines) :

![alt text][lane_area]

#### Center of car

As bottom 2 corners of lane area define where the car is, the bias bettween mid point of those 2 points and mid point of image multiplying by a factor gives the bias of car to the center of road.

### Perspective Transform

In the perspective transforming step, source points of images are essential to know as they define a rectangular area in real world. Luckily, lane lines are parallel in real world so the lane area obtained above is approximately a retangular area in real world. I use points of corners of lane area as source points and arbitray other 4 points defining a retangle area in image as destined points. Here is the warped image:

![alt text][warp_img]

### Polynomial Fitting

Remember the masking step along with Hough Lines Detection Algorithm, there are not much noise except two lane lines in the image. For the sake of simplicity, we can approximate polynomial just based on pixels on the left or right to fit the lane lines respectively. But to be more robust, I use convolution technique to find the lane lines. The convolution technique is that, given a convoltion matrix -- a 2D matrix, it slides horizontally and convolve with the sliding area. The higher value the sliding area has, the more possible it contains a lane line. After convolution and fitting process, here is the output:

![alt text][poly_warp]

### Curvature Calculation

In the previous step, I have fitted the polunomials and have polynomial coefficients now. And thus I can calculate the curvature based on that. Polynomials are not perfectly parallel. After curvature of left and right lane lines been calculated, I give the curvature whose lane is more solid (left lane) a bigger weight and the other less to calculate the overall curvature. 

**NOTE: The radius of curvature should be VERY large (theorectically positive infinity) in frames where the car is driving in a straight lane road. So if the radius of curvature is so large, please note that whether the car is driving in a cruve lane road or not.**

```python
img:test1.jpg 's curvature:1562.27 m
```

### Result

![alt text][final_img]

This image is the result of processed image. 
The **video** is [here](https://github.com/DonaldRR/AdvancedFindLanes/blob/master/processed_project_video.mp4).


### Discussion

#### Issue 1: The lane lines are not still detectable if the light is so bright on the road.

#### Issue 2: ROI is hard coded is this case and it is not applicable in other cases when the lanes are very curved. 

#### Issue 3: Alongside lane lines, there might be other lines have the same direction as the lane lines.

[origin_chessboard]:./write_up_images/origin_chessboard.png "Original Chessboard"
[draw_chessboard]:./write_up_images/draw_chessboard.png "Draw Chessboard"
[undistort_chessboard]:./write_up_images/undistorted_chessboard.png "Undistort Chessboard"
[undistort road]:./write_up_images/undistort_test1.jpg "Undistort Road"
[color_thresh]:./write_up_images/color_thresh_test1.jpg "Color Threshold"
[color_enhance]:./write_up_images/color_enhance_test1.jpg "Color Enhance"
[mag_gradient]:./write_up_images/mag_test1.jpg "Magnitute Gradient"
[x_gradient]:./write_up_images/sobelx_test1.jpg "X Gradient"
[y_gradient]:./write_up_images/sobely_test1.jpg "Y Gradient"
[dir_gradient]:./write_up_images/dir_test1.jpg "Direction Gradient"
[combo]:./write_up_images/combo_test1.jpg "Combination"
[region_mask]:/write_up_images/region_mask_test1.jpg "Region Masking"
[hough_line]:/write_up_images/hough_mask_test1.jpg "Hough Line"
[lane_area]:/write_up_images/lane_area_test1.jpg "Lane Area"
[warp_img]:/write_up_images/warp_test1.jpg "Warp Image"
[poly_warp]:./write_up_images/poly_warp_img_test1.jpg "Poly Warp Image"
[final_img]:./write_up_images/final_test1.jpg "Final Image"
