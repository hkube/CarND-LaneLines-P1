# Udatcity carND P1
#
# Image processing makepipeline
# (c) Harald Kube <harald.kube@gmx.de

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import os
import P1_Helper
from mpl_toolkits.axes_grid1 import ImageGrid

imgPane = plt.figure(1, (10., 10.))
grid = None
gridIndex = 0

def plotPrepare(count):
    global grid
    grid = ImageGrid(imgPane, 111, nrows_ncols=(len(testImages), 2), axes_pad=0.1)
   

def plotImage(image, cmap=None):
    global gridIndex
    grid[gridIndex].imshow(image)
    gridIndex += 1
    

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



inputImageFolder="test_images"
outputImageFolder="test_images_output"
if not os.path.exists(outputImageFolder):
    os.mkdir(outputImageFolder)

testImages = os.listdir(inputImageFolder)
plotPrepare(len(testImages))

for imgName in testImages:
    imgFilePath = '/'.join([inputImageFolder, imgName])
    print(imgFilePath)
    image = mpimg.imread(imgFilePath)
#    print("The image", imgName, "is", type(image), "with dimensions", image.shape)
    plotImage(image)
    
    result=process_image(image)
    plotImage(result)
    
    outFilePath = '/'.join([outputImageFolder, imgName])
    cv2.imwrite(outFilePath, cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
#    break

plt.show()   

print('Done')