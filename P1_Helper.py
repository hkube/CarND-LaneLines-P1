import math
import numpy as np
import cv2

roiTop = 0

def setRoiTop(value):
    """Return the dimensions of the region of interest (top, topLeft, and topRight) """
    global roiTop
    roiTop = value

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
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
    if False:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, 5) #thickness)
        return

    global roiTop
    maxInvSlope = 3
    binSize = 0.2
    ymax = img.shape[0]
    histPos = [ [] for _ in range(int(maxInvSlope/binSize)) ]
    histNeg = [ [] for _ in range(int(maxInvSlope/binSize)) ]
    binloadPos = [ 0 for _ in range(len(histPos))]
    binloadNeg = [ 0 for _ in range(len(histNeg))]

    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2 != x1:
                # Calculate the slope and the offset (the intersection with the bottom of the image)
                m = (y1-y2)/(x2-x1)
                if abs(m) > 0.29:
                    x0 = x1 - (ymax-y1)/m
                else:
                    continue
            else:
                # In case of vertical lines use fixed values to prevent division by zero!
                m = 1e9
                x0 = x1
            binNum = int(1/abs(m)/binSize)
            if m >= 0:
                if binNum >= 0 and binNum < len(binloadPos):
                    # Put the line into the corresponding bin
                    histPos[binNum].append((m, x0))
                    # Increment the bin load counter
                    binloadPos[binNum] += 1
            else:
                if binNum >= 0 and binNum < len(binloadNeg):
                    # Put the line into the corresponding bin
                    histNeg[binNum].append((m, x0))
                    # Increment the bin load counter
                    binloadNeg[binNum] += 1
#            cv2.line(img, (x1, y1), (x2, y2), [0, 255, 0], 2)

    # Find the bin in the histogram of positive slopes with the highest load
    mval, midx = max([(v, i) for i,v in enumerate(binloadPos)])

    lines2 = []
    if mval > 0:
        # Calculate the average slope and offset of the line segment in the bin with the highest load and its neighbor bins
        lineSegments = histPos[midx]
        if midx > 0:
            lineSegments += histPos[midx-1]
        if midx < (len(histPos)-1):
            lineSegments += histPos[midx+1]
        mMean = np.mean([m for m,x0 in lineSegments])
        x0Mean = np.mean([x0 for m,x0 in lineSegments])
        lines2.append((mMean, x0Mean))


    # Search the bin with the second highest number of line segments
    mval, midx = max([(v, i) for i,v in enumerate(binloadNeg)])

    if mval > 0:
        # Calculate the average slope and offset of the line segment in the bin with the second highest load and its neighbor bins
        lineSegments = histNeg[midx]
        if midx > 0:
            lineSegments += histNeg[midx-1]
        if midx < (len(histNeg)-1):
            lineSegments += histNeg[midx+1]
        mMean = np.mean([m for m,x0 in lineSegments])
        x0Mean = np.mean([x0 for m,x0 in lineSegments])
        lines2.append((mMean, x0Mean))

    for m,x0 in lines2:
        y1 = int(img.shape[0])
        x1 = int(x0)
        y2 = int(roiTop)
        x2 = int(x0 + (ymax-y2)/m)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)