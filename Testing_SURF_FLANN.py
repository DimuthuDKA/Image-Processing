from datetime import datetime
start_time = datetime.now()
import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
img1 = cv.imread('10.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('4 -Nodistortion.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
surf = cv.xfeatures2d.SURF_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
#ratio test as per Lowe's paper
counter=1
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[0,1]
    if m.distance < 0.7*n.distance:
        counter=counter+1
        
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags=4)
print(counter)
#draw_params = dict(matchColor=(255,0,255),singlePointColor=(255,0,0),flags=4)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3)
plt.title('Defected Image 10-SURF Descriptor & FLANN Matcher with Good Matcher Points')
plt.xlabel('X-Pixels')
plt.ylabel('Y-Pixels')
plt.show()

