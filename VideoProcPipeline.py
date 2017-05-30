# Udatcity carND P1
#
# Image processing makepipeline
# (c) Harald Kube <harald.kube@gmx.de

#importing some useful packages
import P1_Helper
import numpy as np
import cv2
#import os
#from ImageProcPipeline import process_image

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def process_image(image):
    imgGray = P1_Helper.grayscale(image)
    imgBlur = P1_Helper.gaussian_blur(imgGray, 9)     
    imgCanny = P1_Helper.canny(imgBlur, 50, 150)
#    plotImage(imgCanny, cmap="gray")
    
    maxX = imgGray.shape[1]-1
    maxY = imgGray.shape[0]-1
    imageMask = np.array([[[0, maxY], [maxX/2-20, maxY/2+40], [maxX/2+20, maxY/2+40], [maxX, maxY], [0, maxY]]], dtype=np.int32)
    imgCrop = P1_Helper.region_of_interest(imgCanny, imageMask)
    
    imgLines = P1_Helper.hough_lines(imgCrop, rho=1, theta=np.pi/180, threshold=10, min_line_len=70, max_line_gap=70)
#    plotImage(imgLines) #, cmap="gray")
    
    result = cv2.add(image, imgLines)
    print("result is", type(image), "with dimensions", image.shape)
    return result


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,2)
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

print('Done')