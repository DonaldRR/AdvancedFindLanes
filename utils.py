import numpy as np
import cv2
import glob
import math
from sklearn.linear_model import LinearRegression
from scipy.cluster.hierarchy import ward, fclusterdata
from scipy.spatial.distance import pdist
import copy
import matplotlib
from matplotlib import pyplot as plt


def cal_curvature_2nd(x, coefs):
    return np.power(1 + np.power((2 * coefs[0] * x + coefs[1]), 2), 1.5) / (2 * np.absolute(coefs[0]))


def measure_curvature_pixels(ploty, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = cal_curvature_2nd(y_eval, left_fit)  ## Implement the calculation of the left line here
    right_curverad = cal_curvature_2nd(y_eval, right_fit)  ## Implement the calculation of the right line here

    return left_curverad, right_curverad

def fit_polynomial(binary_warped):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Find our lane pixels first
    x_size, y_size = binary_warped.shape[1], binary_warped.shape[0]
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx = nonzerox[nonzerox < x_size/2]
    lefty = nonzeroy[nonzerox < x_size/2]
    rightx = nonzerox[nonzerox > x_size/2]
    righty = nonzeroy[nonzerox > x_size/2]
    # leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_fit_m = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_m = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    dir = 1
    if len(leftx) > len(rightx):
        dir = 0

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    output_img = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2RGB)
    output_img[lefty, leftx] = [255, 0, 0]
    output_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    ploty = np.reshape(ploty, (-1, 1))
    left_fitx = np.reshape(left_fitx, (-1, 1))
    right_fitx = np.reshape(right_fitx, (-1, 1))
    cv2.polylines(output_img, np.int32([np.hstack((left_fitx, ploty))]), False, (0, 255, 255), 3)
    cv2.polylines(output_img, np.int32([np.hstack((right_fitx, ploty))]), False, (0, 255, 255), 3)

    return output_img, np.int32(ploty * ym_per_pix), left_fit_m, right_fit_m, dir

def fit_line(pt1, pt2):

    x1, y1, x2, y2 = float(pt1[0]), float(pt1[1]), float(pt2[0]), float(pt2[1])

    try:
        a = (y1 - y2) / (x1 - x2)
    except:
        a = (y1 - y2) / ((x1 - x2) + 0.000001)
    b = y1 - a * x1

    return a, b

def line_dist(p1, p2):

    a1, b1 = p1[0], p1[1]
    a2, b2 = p2[0], p2[1]

    return ((p1[0] + p1[1])/2 - (p2[0] + p2[1])/2)**2 + ((p1[2] + p1[2])/2 - (p2[3] + p2[3])/2)**2

    # return np.abs(b1 - b2) * 10000 + 500 * ((a1/a2)**2 + (a2/a1)**2 - 2)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2, cluster=True):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    Y_l, Y_r, x_l, x_r = [], [], [], []
    left_coefs = []
    right_coefs = []

    # Slopes and Region thresholding
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            a, b = fit_line((y1, x1), (y2, x2))
            # n_copies = int(math.pow(math.sqrt(math.pow((y2 - y1), 2) + math.pow((x2 - x1), 2)) / 20., 2))
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            if abs(a) > 0.5:

                if a < 0 and max(x1, x2) < img.shape[1] / 2:
                    Y_l.append([[y1], [y2]])
                    x_l.append([[x1], [x2]])
                    left_coefs.append([a, b])
                    # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
                elif a > 0 and min(x1, x2) > img.shape[1] / 2:
                    Y_r.append([[y1], [y2]])
                    x_r.append([[x1], [x2]])
                    right_coefs.append([a, b])
                    # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

    Y_l, Y_r, x_l, x_r = np.array(Y_l), np.array(Y_r), np.array(x_l), np.array(x_r)

    n_lostLines = 0
    y1 = int(img.shape[0] * 13/ 20.0)
    y2 = img.shape[0]
    # Clutering and Linear regression
    if cluster:

        # Clustering
        try:
            left_clusts = fclusterdata(np.squeeze(np.hstack((x_l, Y_l)), axis=2), t=2, criterion='maxclust', metric=line_dist)
            right_clusts = fclusterdata(np.squeeze(np.hstack((x_r, Y_r)), axis=2), t=2, criterion='maxclust', metric=line_dist)
            left_idxs = (left_clusts == np.argmax(np.bincount(left_clusts)))
            right_idxs = (right_clusts == np.argmax(np.bincount(right_clusts)))

            new_Y_l, new_Y_r, new_x_l, new_x_r = Y_l[left_idxs], Y_r[right_idxs], x_l[left_idxs], x_r[right_idxs]
            new_Y_l, new_Y_r, new_x_l, new_x_r = np.reshape(new_Y_l, (-1, 1)), np.reshape(new_Y_r, (-1, 1)), np.reshape(new_x_l, (-1, 1)), np.reshape(new_x_r, (-1, 1))
        except:
            print(x_r.shape, x_r)
            print(Y_r.shape, Y_r)
            print(np.hstack((x_r, Y_r)).shape, np.hstack((x_r, Y_r)))
            return False


        # Linear regression
        try:
            new_clf_l = LinearRegression().fit(new_Y_l, new_x_l)
            new_x1_l = int(new_clf_l.predict(y1))
            new_x2_l = int(new_clf_l.predict(y2))
            cv2.line(img, (new_x1_l, y1), (new_x2_l, y2), (255, 255, 255), 30)
        except:
            n_lostLines += 1

        try:
            new_clf_r = LinearRegression().fit(new_Y_r, new_x_r)
            new_x1_r = int(new_clf_r.predict(y1))
            new_x2_r = int(new_clf_r.predict(y2))
            cv2.line(img, (new_x1_r, y1), (new_x2_r, y2), (255, 255, 255), 30)
        except:
            n_lostLines += 1

        return [[new_x1_l, y1], [new_x1_r, y1], [new_x2_r, y2], [new_x2_l, y2]]
    else:

        Y_l, Y_r, x_l, x_r = np.reshape(Y_l, (-1, 1)), np.reshape(Y_r, (-1, 1)), np.reshape(x_l, (-1, 1)), np.reshape(x_r, (
        -1, 1))

        # Linear regression
        try:
            clf_l = LinearRegression().fit(Y_l, x_l)
            x1_l = int(clf_l.predict(y1))
            x2_l = int(clf_l.predict(y2))
            cv2.line(img, (x1_l, y1), (x2_l, y2), (255, 255, 255), 30)
        except:
            n_lostLines += 1

        try:
            clf_r = LinearRegression().fit(Y_r, x_r)
            x1_r = int(clf_r.predict(y1))
            x2_r = int(clf_r.predict(y2))
            cv2.line(img, (x1_r, y1), (x2_r, y2), (255, 255, 255), 30)
        except:
            n_lostLines += 1

        return [[x1_l, y1], [x1_r, y1], [x2_r, y2], [x2_l, y2]]


def redraw_lines(img, lines, low_slope_threshold=0.5, high_slope_threshold=0.6):
    for line in lines:
        for x1, y1, x2, y2 in line:
            _slope = abs(float((y2 - y1)) / (x2 - x1))
            if _slope < low_slope_threshold:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
            #             elif _slope > high_slope_threshold:
    #                 cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """

    # Get original lines
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    # for t in range(2):
    #     img = redraw_lines(img, lines)
    #     lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
    #                             maxLineGap=max_line_gap)
    line_img= np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    pts = draw_lines(line_img, lines, cluster=False)

    return line_img, pts
    line_img = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)

    if pts == False:
        return img, pts

    # Get lines from masked image
    mask_img = np.zeros_like(img)
    mask_img[(img == 1) & (line_img == 1)] = 1
    new_lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    new_line_img= np.zeros((mask_img.shape[0], img.shape[1], 3), dtype=np.uint8)
    new_pts = draw_lines(new_line_img, new_lines, cluster=False)


    return new_line_img, new_pts


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient is 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient is 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    absolute = np.absolute(sobel)
    scaled = np.uint8(255 * np.float32(absolute) / np.max(absolute))
    binary = np.zeros_like(absolute)
    binary[(scaled > thresh_min) & (scaled < thresh_max)] = 1
    binary_output = binary

    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    absolute = np.absolute(sobel)
    scaled = np.uint8(255 * np.float32(absolute) / np.max(absolute))
    binary = np.zeros_like(scaled)
    binary[(scaled > mag_thresh[0]) & (scaled < mag_thresh[1])] = 1
    binary_output = binary

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobelx[sobelx == 0] += 0.00001
    degree = np.absolute(np.arctan(sobely / sobelx))
    binary = np.zeros_like(degree)
    binary[(degree > thresh[0]) & (degree < thresh[1])] = 1
    binary_output = binary

    return binary_output


def hls_select(img, thresh1=(0, 255), thresh2=(0, 255), thresh3=(0, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result

    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls_img[:, :, 0]
    l_channel = hls_img[:, :, 1]
    s_channel = hls_img[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[((h_channel > thresh1[0]) & (h_channel <= thresh1[1])) &
                  ((l_channel > thresh2[0]) & (l_channel <= thresh2[1])) &
                  ((s_channel > thresh3[0]) & (s_channel <= thresh3[1]))] = 1

    return binary_output



def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped