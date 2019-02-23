import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
UBIT = "BASKARAD"
np.random.seed(sum([ord(c) for c in UBIT]))

sift = cv2.xfeatures2d_SIFT.create()
img1 = cv2.imread("tsucuba_left.png")
img1_bw = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread("tsucuba_right.png")
img2_bw = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

""" Plotting the Keypoints on the input images"""
kp1 = sift.detect(img1,None)
img1withkp=cv2.drawKeypoints(img1,kp1,outImage=np.array([]))
cv2.imwrite('task2_sift1.jpg',img1withkp)

kp2 = sift.detect(img2,None)
img2withkp=cv2.drawKeypoints(img2,kp2,outImage=np.array([]))
cv2.imwrite('task2_sift2.jpg',img2withkp)


kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

"""draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)"""

draw_params = dict(matchColor = (0,0,255),
                  singlePointColor = (255,0,0),
                  matchesMask = matchesMask,
                  flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
cv2.imwrite('task2_matches_knn.jpg',img3)

### Fundamental matrix
good = []

pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)

print("Fundamental Matrix")
print(F)

####

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

#taking randomly 10 samples and matching it
random_points = np.random.randint(0, len(pts1), 10)


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

selected_point1,selected_point2 = list(), list()
for i, (p1, p2) in enumerate(zip(pts1, pts1)):
    if i in random_points:
        selected_point1.append(p1)
        selected_point2.append(p2)
selected_point1 = np.float32(selected_point1)
selected_point2 = np.float32(selected_point2)
colors = []
for i in range(0,10):
    colors.append(tuple(np.random.randint(0,255,3).tolist()))

img1lines = cv2.computeCorrespondEpilines(selected_point1.reshape(-1, 1, 2), 2, F)
img1lines = img1lines.reshape(-1, 3)
img1lines1,image1_lines2 = drawlines(img1_bw,img2_bw,img1lines,selected_point1,selected_point2)

img2lines = cv2.computeCorrespondEpilines(selected_point2.reshape(-1, 1, 2), 2, F)
img2lines = img2lines.reshape(-1, 3)
img2lines1,image2_lines2 = drawlines(img2_bw,img1_bw,img2lines,selected_point2,selected_point1)

cv2.imwrite('task2_epi_right.jpg',img2lines1)
cv2.imwrite('task2_epi_left.jpg',img1lines1)

imgL = cv2.imread('tsucuba_left.png',0)
imgR = cv2.imread('tsucuba_right.png',0)

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(imgL,imgR)
cv2.imwrite('task2_disparity.jpg',disparity)

plt.imshow(disparity,'gray')
plt.show()


""" https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective"""

"""https://github.com/techfort/pycv/blob/master/chapter4/depth.py"""
