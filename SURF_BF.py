from datetime import datetime
start_time = datetime.now()
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('4 -Nodistortion.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('3 - Copy.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
surf = cv.xfeatures2d.SURF_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
print(len(matches))
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.

#draw_params = dict(matchColor=(0,0,255),singlePointColor=(255,0,0),flags=4)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=4)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
plt.imshow(img3)
plt.title('SURF Descriptor & BF Matcher With All Feature Points')
plt.xlabel('X-Pixels')
plt.ylabel('Y-Pixels')
plt.show()
cv.destroyAllWindows()
