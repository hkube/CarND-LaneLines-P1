# **Finding Lane Lines on the Road**

## Writeup

---

### Reflection

### 1. Description of my pipeline

My pipeline consisted of 11 steps.

1. I converted the images to grayscale
2. I applied a gaussian blur to the image
3. I applied a canny filter to the image to detect the edges
4. I applied a mask to the image so that only the edges in the region of interest remain and the edges outside are deleted
5. I applied the hough algorithm to detect all lines in the images
6. I calculated the slope and the intersection with the bottom of the image for each of the detected lines
7. I separated the lines with a negative slope from the line with the positive slope in separate line sets
8. I selected the lines with commonest slopes in both lines sets and calculated the mean slope and intersection to get the left and the right lane line
9. I calculated the average slope and intersection of the lane lines in the current image and up to four previous images to reduce the jitter of the lane lines
10. I draw the lane lines from bottom of the image to the top of the region of interest into an empty image
11. I added the image with the lane lines as an overlay to the original image

The steps 6 to 10 are the modification of the draw_lines() function in order to draw a single line on the left and right lanes.


### 2. Potential shortcomings with my current pipeline

One potential shortcoming would be what would happen when the car changes from one to another lane and the lane line is
in the center of the image. Due to the separation of the detected lines into two sets, one with lines with a positive slope
and one with lines with a negative slope could lead to the effect that the line in the center is not correctly detected

Another shortcoming could be that the angle between the left and right lane line is not checked to be within an expected range.


### 3. Possible improvements to my pipeline

A possible improvement would be to use a weighted arithmetic means or the median to calculate the average of the slopes and intersections.

Another potential improvement could be to discard lines found by the hough algorithm which are not at the border of white or yellow regions.

Another potential improvement could be to check that the angle between the left an the right lane line is within an expected range.
