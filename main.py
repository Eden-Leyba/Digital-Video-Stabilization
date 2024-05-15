import numpy as np
import cv2

# Global variables
SCALING = 1.2
SMOOTHING_RADIUS = 50

def fix_border(_frame):
    s = _frame.shape
    # Scale the image without moving the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, SCALING)
    _frame = cv2.warpAffine(_frame, T, (s[1], s[0]))
    return _frame

# Smooth the trajectory
def moving_average(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius,radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

def smooth(_trajectory, smooth_radius):
    _smoothed_trajectory = np.copy(_trajectory)
    for i in range(3):
        _smoothed_trajectory[:,i] = moving_average(trajectory[:, i], radius=smooth_radius)
    return _smoothed_trajectory


# Step 1: Set Input and Output Video
# input video
capture = cv2.VideoCapture('video.mp4')

n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec for output video
# codec = computer program that encodes or decodes a data stream
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# get fps
fps = capture.get(cv2.CAP_PROP_FPS)

# output video setup - we'll do it later

# Step 2: Read first frame and convert to grayscale
# Read first frame
_, prev = capture.read()

# Convert previous frame from RGB to Grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Step 3: Find motion between frames
"""
    Iterate Over all the frames
    Find motion between the current and the previous frame
    The Euclidean motion model requires that we know the motion of only 2 points in the two frames
    In practice, we will find the motion of 50-100 points, and use them to make a better estimation of the motion model
"""

# creates np array, rows = np.frames-1, columns = 3 (dx,dy,da)
transforms = np.zeros((n_frames - 1, 3), np.float32)

# Loop through all frames
for i in range(n_frames - 2):
    # goodFeatureToTrack detects feature points in previous frame who are good for tracking
    prev_pts = (
        cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3))

    # Read next frame
    success, curr = capture.read()
    if not success:
        break

    # Convert frame curr from RGB to Grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Use the Lucas-Kanade algorithm to calculate the optical flow
    curr_pts, status, err = (
        cv2.calcOpticalFlowPyrLK(prevImg=prev_gray, nextImg=curr_gray, prevPts=prev_pts, nextPts=None))

    # Sanity check
    assert prev_pts.shape == curr_pts.shape

    # Filter only valid points
    index = np.where(status == 1)[0]
    """
        From the calcOpticalFlowPyrLk doc:
        status output status vector: each element of the vector is set to 1 if
        the flow for the corresponding features has been found, otherwise, it is set to 0.
        
        In other words, status == 1 <==> the point is valid
    """
    prev_pts = prev_pts[index]
    curr_pts = curr_pts[index]

    # Find transformation matrix
    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_
                                       
    # Extract translation
    dx = m[0, 2]
    dy = m[1, 2]

    # Extract rotation angle
    da = np.arctan2(m[1, 0], m[0, 0])

    transforms[i] = [dx, dy, da]

    # Move to the next frame
    prev_gray = curr_gray

    print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))


# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)  # sum over rows for each of the 3 columns

smoothed_trajectory = smooth(trajectory, SMOOTHING_RADIUS)

# Calculate smooth transforms
difference = smoothed_trajectory - trajectory
transforms_smooth = transforms + difference

# Apply smoothed camera motion to frames

# First we reset the stream to the first frame
capture.set(cv2.CAP_PROP_POS_FRAMES,0)

# output video object, set it to None for now and initialize it in the first iteration
out = None

for i in range(n_frames-2):
    success, frame = capture.read()
    if not success:
        break

    # Extract transformations from the smoothed transformations array
    dx = transforms_smooth[i,0]
    dy = transforms_smooth[i,1]
    da = transforms_smooth[i,2]

    # Reconstruct the transformation matrix (מטריצה מייצגת) of the smoothed transformations array
    A = np.zeros((2,3), np.float32)
    A[0,0] = np.cos(da)
    A[0,1] = -np.sin(da)
    A[1,0] = np.sin(da)
    A[1,1] = np.cos(da)
    A[0,2] = dx
    A[1,2] = dy

    # Apply affine wrapping to the given frame
    frame_stabilized = cv2.warpAffine(frame,A,(w,h))

    # Fix border artifacts
    frame_stabilized = fix_border(frame_stabilized)

    # Write the frame to the file
    frame_out = cv2.hconcat([frame, frame_stabilized])
    width,height = int(frame_out.shape[1]/2), int(frame_out.shape[0]/2)
    if out is None:
        # output video setup
        out = cv2.VideoWriter('video_out.mp4', fourcc, fps, (width,height))

    # If the image is too big, resize it.
    if frame_out.shape[1] > 1920:
        frame_out = cv2.resize(frame_out, (int(frame_out.shape[1]/2), int(frame_out.shape[0]/2)))
        # frame_out = cv2.resize(frame_out, (w,h))

    cv2.imshow("Before and After", frame_out)
    cv2.waitKey(10)
    out.write(frame_out)

capture.release()
out.release()
cv2.destroyAllWindows()
