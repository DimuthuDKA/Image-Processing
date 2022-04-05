# Brute-Force Matching with ORB Descriptors
import numpy as np
from datetime import datetime
start_time = datetime.now()
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('4 -Nodistortion.jpg',0)          # queryImage
img2 = cv2.imread('3 - Copy.jpg',0) # trainImage

# Initiate ORB detector
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


n=50 # Draw first n matches.

draw_params = dict(singlePointColor=(0, 0, 255),matchColor=(255, 0, 0),flags=4)
img3 =cv2.drawMatches(img1,kp1,img2,kp2,matches[:len(matches)],None, **draw_params)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

#y=plt.figure(figsize=(100,100));
plt.imshow(img3),
plt.title('ORB Descriptor & BF Matcher with All Feature Points')
plt.xlabel('X-Pixels')
plt.ylabel('Y-Pixels')
plt.show()
cv.destroyAllWindows()
plt.show();


