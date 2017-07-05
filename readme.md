## [Advanced Lane Finding Project](https://github.com/udacity/CarND-Advanced-Lane-Lines)

![Alt Text](./project_video_result.gif)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard_corners.png "chessboard corners"
[image2]: ./output_images/undistorted_chessboard.png "undistorted chessboard"
[image3]: ./output_images/undistorted_test1.png "undistorted test1"
[image4]: ./output_images/color_thresholding.jpg "color spaces"
[image5]: ./output_images/binary_thresholding.jpg "binary thresholding"
[image6]: ./output_images/warped.jpg "warped"
[image7]: ./output_images/sliding_window.jpg "sliding window"
[image8]: ./output_images/result_test1.jpg "result"
[image9]: ./output_images/result_testimgs.png "results"


---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook "main.ipynb". 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.
![alt text][image1]

`undistort()` takes a test image, converts it to grayscale, computes the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function, `objpoints` and `imgpoints`.  It then applies a distortion correction to the image using the `cv2.undistort()` function.
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Undistortion applied to one of the test images:

![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used only color thresholds with a region mask to generate a binary image. `binary_thresh()` combines H, L, S of the HLS color space and B of the LAB color space to output a binary image. Since the saturation channel was prone to change in brightness, hue was ANDed with saturation to make it more robust

![alt text][image4]

Gradient thresholding with the sobel operator added noise and wasn't too helpful. The final thresholding was 

`(S & H) | B | L`

A region mask was applied to discard other information such as the trees and sky. 
Here's the final combination on a test image:

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

`perspective_transform()` takes in an input image, `src` and `dst` points and returns a warped image. The src and dst points were hardcoded by eyeballing a sample image.

```python
src = np.float32([[577, 462], [705, 462], [1053, 685], [254, 685]])
dst = np.float32([[offset, 0], [w-offset, 0], [w-offset, h], [offset, h]])
```

An example of an undistorted image and its warped counterpart:

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

`find_lanes_sliding_window()` detects lane lines and fits a polynomial. It takes a binary warped image as an input, takes a histogram of the lower half of the image. The peak of the left and right bottom half is the starting point of the left and right lane respectively. Two sliding windows follow these pixels to the top of the image. A second order polynomial fits these pixels to form the left and right lane line. 

![alt text][image7]

Once we've established valid lane lines in one frame, `find_lanes_prev_fit()` takes the binary warped image of the next frame and the lane lines found in the previous fit. It restricts the search span within a margin of the previously detected lanes and fits a polynomial to the pixels detected within that margin. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

`get_radius_curvature_center_offset()` takes the warped image, left and right lane pixels, scales them to meters, fits a second order polynomial to them, returns the [radius of curvature](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) and offset of the vehicle from the center of the lane. It assumes that the camera is positioned at the center of the vehicle. 

Radius of curvature:
`rad_curv = ((1 + (2*fit_m[0]*y_eval + fit_m[1])**2)**1.5) / np.absolute(2*fit_m[0])`

Offset from center:
```python
lane_center = (left_bottomxm + right_bottomxm)/2
vehicle_pos = (img_warped.shape[1]/2) * xm_per_pix
offset = lane_center - vehicle_pos 
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

`get_binary_warped_img()` combines some of the steps above. It takes in a RGB image and returns a binary warped image and the inverse perspective transform matrix. 
`draw_lane()` draws the lane lines found in `find_lanes_sliding_window()` or `find_lanes_prev_fit()` and then unwarps the image.

Here is an example of my result on a test image:

![alt text][image8]

Pipeline on the test images:

![alt text][image9]

For a video, lane lines are detected using `find_lanes_sliding_window()`. We use `find_lanes_prev_fit()` only if the previous lane lines are valid. Lane lines are valid if the mean distance between them is about 480px. If the lane lines are invalid, we resort to the previously detected lanes. The lane lines in the last 10 frames are averaged to smooth out outliers and prevent jerks. 

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_result.mp4)

Here's a [link to my result on the optional challenge](./challenge_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue was playing around with gradients and color spaces to find the right balance of thresholds. The HLS space and the B channel of LAB worked quite well for the test images and the challenge video. The saturation channel of HLS gave decent results for the yellow and white lanes, although it was weak and prone to noise. The higher end of the B channel of the LAB space clearly filtered the yellow lane. Later when trying to fit a polynomial, I found out that the white lines in some images weren't very clear and was resulting in abnormal polynomial fits. The luminosity channel reinforced the white lanes. It would be desirable to discriminate the lanes better, especially in adverse conditions.

In cases where one lane line is detected well (an extrapolated version of the previous fits) but the other lane is not valid, we can predict where the other lane will be since we know the distance between the two lanes. This will be helpful for sharp turns. Even if one lane is detected, the other can be predicted to be parallel to it. The current implementation ignores lane even when one lane is not detected and resorts to an average of the previous n fits. This might fail on sharp turns where we don't have a decent detection of one lane. 
On failure to detect both lanes, we can extrapolate the previous fits and use that for some frames ahead depending on the speed of the vehicle until we get a valid detection.

One improvement would be to use a weighted average for smoothing the previous n fits instead of just a simple average. The weights can be a measure of how confident we are about the fit. One metric would be the mean distance between the lanes.
