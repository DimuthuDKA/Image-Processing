import cv2
import numpy as np
MIN_MATCH_COUNT=30                  # The minimum nuber of key points that should be matching to be considered a frame which consists the image you are searching

detector=cv2.xfeatures2d.SIFT_create() # defined the detector , you can change to surf , orb if required 

FLANN_INDEX_KDITREE=0                  # these are parameters of Flann Matcher 
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

trainImg=cv2.imread('capture.jpg',0)    #save the image to be compared in the same folder and add its name instad of capture
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)#detect features of saved image in the ROM

cam=cv2.VideoCapture(0)# Define Camera Input , change Variable 0 to 1 or 2 if camera not recognized
while True:
    ret, QueryImgBGR=cam.read()             #reading the camera frame
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)  #convert to black and white
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None) # Detect Features of Captured image
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)   #match the features and save results in an array called "matches"
    
    #matches varible calculated above is actually a Matrix representing the comparative distances between key points.
    #only distances which are nearly in ratio of  0.75 to actual distances are considered as good matching points 
    goodMatch=[]
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)#tp and tq arrays are needed to calculate the homography matrix which is used to transform image ( just like Transform matrix)
        h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H) 
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)# draw a border on the captured frame according to calculated homography matrix
        
    cv2.imshow('result',QueryImgBGR)          # display the new image after drawing the oultine if there is a homography matrix
    print ("Matches : ",str((len(goodMatch))))# displays number of matches detected seperately for each frame 

    if cv2.waitKey(10)==ord('q'):             # Exit screen when pressing q on key pad 
        break
cam.release()
cv2.destroyAllWindows()

