import cv2
import numpy as np
import time

def movingAverage(curve, radius):
  window_size = 2 * radius + 1
  # Define the filter
  f = np.ones(window_size)/window_size
  # Add padding to the boundaries
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
  # Apply convolution
  curve_smoothed = np.convolve(curve_pad, f, mode='same')
  # Remove padding
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed

def smooth(trajectory):
  smoothed_trajectory = np.copy(trajectory)
  SMOOTHING_RADIUS = 50
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
 
  return smoothed_trajectory

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

cap = cv2.VideoCapture('vid03.mp4')

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

    #show points
    # for pt in curr_pts:
    #     #print(pt[0])
    #     curr_frame = cv2.circle(curr_frame,(int(pt[0][0]),int(pt[0][1])),radius=0,color=(255,0,0),thickness=10)
    #transformation matrix
    m = cv2.estimateAffinePartial2D(prev_pts,curr_pts)
    #m = cv2.estimateRigidTransform(prev_pts,curr_pts,fullAffine=False)

    #m = cv2.invertAffineTransform(m[0])

    # frame_out = cv2.warpAffine(curr_frame, m[0], (w,h))

    # #frame_out = cv2.hconcat([curr_frame,frame_out])
    # frame_out = cv2.addWeighted(curr_frame,0.5,frame_out,0.5,0)
    # if frame_out.shape[1] > 1920:
    #      frame_out = cv2.resize(frame_out, (int(frame_out.shape[1]/2), int(frame_out.shape[0]/2)));

    # cv2.imshow('before - after',frame_out)
    # cv2.waitKey(10)

    #translation
    dx = m[0][0,2]
    dy = m[0][1,2]

    #rotation angle
    da = np.arctan2(m[0][0,1],m[0][0,0])

    #store transformation
    transforms[f] = [dx,dy,da]

    #move to next frame
    prev_gray = curr_gray    

    print("Frame: " + str(f) +  "/" + str(nFrames) + " -  Tracked points : " + str(len(prev_pts)))
    

    #trajectory
trajectory = np.cumsum(transforms,axis=0)
smooth_trajectory = smooth(trajectory)

difference = smooth_trajectory - trajectory
transforms_smooth = transforms + difference



    #cv2.imshow('frame',curr_frame)
    #cv2.imshow('img0',img0)
    #cv2.waitKey(1)
    #out.write(img0)

cap.set(cv2.CAP_PROP_POS_FRAMES,0)

for i in range(nFrames-2):
    success, frame = cap.read()
    if not success:
        break

    dx = transforms_smooth[i,0]
    dy = transforms_smooth[i,1]
    da = transforms_smooth[i,2]

    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2,3), np.float32)
    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy

    # Apply affine wrapping to the given frame
    frame_stabilized = cv2.warpAffine(frame, m, (w,h))

    # Fix border artifacts
    frame_stabilized = fixBorder(frame_stabilized) 
    out.write(frame_stabilized)

    frame_out = cv2.hconcat([frame, frame_stabilized])

    # If the image is too big, resize it.
    if frame_out.shape[1] > 1920:
        frame_out = cv2.resize(frame_out, (int(frame_out.shape[1]/2), int(frame_out.shape[0]/2)));
    
    
    #frame_out = cv2.addWeighted(frame,0.5,frame_stabilized,0.5,0)
    cv2.imshow('before-after',frame_out)
    #cv2.imshow('after',frame_stabilized)
    cv2.waitKey(10)
   

out.release()


