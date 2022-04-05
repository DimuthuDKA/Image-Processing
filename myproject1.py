
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt 

img1 = cv2.imread('4 -Nodistortion.jpg',0)# qu.eryImacge
img2 = cv2.imread('4 -distortion2.jpg',cv2.IMREAD_GRAYSCALE) # trainImage

#cv2.imshow('Image1', img1)
#cv2.imshow('Image2', img2)
#cv2.waitKey(0)
plt.figure(1)
plt.subplot(211)
plt.imshow(img1,cmap='gray')

plt.subplot(212)
plt.imshow(img2,cmap='gray')
plt.show()


# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)
print(len(matches))

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=4)
plt.imshow(img3),plt.show()
cv2.imshow('image',img3)  
  
# Maintain output window utill 
# user presses a key 
cv2.waitKey(0)
cv2.destroyAllWindows()

