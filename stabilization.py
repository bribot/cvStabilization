import cv2
import numpy as np
import time

cap = cv2.VideoCapture('vid0.mp4')

nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

#CODEC
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#output
out = cv2.VideoWriter('vid0_out.mp4',fourcc,fps,(w,h))

#transformation
transforms = np.zeros((nFrames-1,3),np.float32)

ret,frame = cap.read()
prev_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

for f in range(nFrames):
    #detect features
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,maxCorners=200,qualityLevel=0.01,minDistance=30,blockSize=3)

    #next frame
    success, curr_frame = cap.read()
    if not success:
        break

    #grayscale
    curr_gray = cv2.cvtColor(curr_frame,cv2.COLOR_BGR2GRAY)

    #optical flow
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,curr_gray,prev_pts,None)

    #sanity check
    assert prev_pts.shape == curr_pts.shape

    #filter calid points
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    #transformation matrix
    m = cv2.estimateRigidTransform(prev_pts,curr_pts,fullAffine=False)

    #translation
    dx = m[0,2]
    dy = m[1,2]

    #rotation angle
    da = np.arctan2(m[0,1],m[0,0])

    #store transformation
    transforms[i] = [dx,dy,da]

    #move to next frame
    prev_gray = curr_gray

    print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

out.release()


