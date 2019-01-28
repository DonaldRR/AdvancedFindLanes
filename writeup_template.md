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

```python
img:test1.jpg 's curvature:1562.27 m
```

### Result

![alt text][final_img]

This image gives the result of processed image. The video is here.



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

### 

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
