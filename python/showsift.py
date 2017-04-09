import cv2
import numpy as np
import sys
import random
import math

def readpoints(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    points = np.array(map(float, lines[-1].split()))
    # conver to 0-indexed
    points = points - 1
    return np.reshape(points, (-1, 2))

img = cv2.imread(sys.argv[1])
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

points = readpoints(sys.argv[2])
print points, points.shape

sift = cv2.xfeatures2d.SIFT_create()

kp = []
npoints, _ = points.shape
for i in range(1):
    kp.append(cv2.KeyPoint(points[i, 0], points[i, 1], 8., random.random()*360, 0., 0, -1))
#kp, desc = sift.compute(gray, kp)
kp, desc = sift.detectAndCompute(gray, None)
print desc, np.linalg.norm(desc[0])

img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)
