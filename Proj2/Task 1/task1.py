import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
UBIT = "BASKARAD"
np.random.seed(sum([ord(c) for c in UBIT]))

#sift = cv2.SIFT()

sift = cv2.xfeatures2d_SIFT.create()
img1 = cv2.imread("mountain1.jpg",0)
img2 = cv2.imread("mountain2.jpg",0)
"""cv2.imshow("Oct11",img1)
cv2.waitKey(0)"""

""" Plotting the Keypoints on the input images"""
kp1 = sift.detect(img1,None)
img1withkp=cv2.drawKeypoints(img1,kp1,outImage=np.array([]))
cv2.imwrite('task1_sift1.jpg',img1withkp)

kp2 = sift.detect(img2,None)
img2withkp=cv2.drawKeypoints(img2,kp2,outImage=np.array([]))
cv2.imwrite('task1_sift2.jpg',img2withkp)

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
cv2.imwrite('task1_matches_knn.jpg',img3)


#matchesMask = []
### To get the HOMOGRAPHY matrix M
MIN_MATCH_COUNT = 10
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None


print("M")
print(M)

"""print("matchesMask")
print(matchesMask)"""
"""
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
"""

#taking randomly 10 samples and matching it
good10 = random.sample(good, 10)



if len(good10)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good10 ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good10 ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    #print ("Not enough matches are found - %d/%d" % (len(good10),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good10,None,**draw_params)
cv2.imwrite('task1_matches.jpg',img3)
#plt.imshow(img3, 'gray'),plt.show()

h,w = img1.shape
#### Pano

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img2
    return result

#dst_pts = float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
#src_pts = float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
#M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

print("M")
print(M)
img1 = cv2.imread("mountain1.jpg")
img2 = cv2.imread("mountain2.jpg")
result = warpTwoImages(img1, img2, M)
cv2.imwrite('task_pano.jpg',result)

"""
#watch out from M in the end. Use appropriate M
img_stitch = []
dst = cv2.warpPerspective(img2,M,(w*2,h))
cv2.imwrite('dst.jpg',dst)
"""
"""
h,w = img1.shape
h2,w2 = img2.shape
img_stitch = []
warpPerspective(img2, img_stitch, M, Size(w2*2, h2));
half = img_stitch(Rect(0, 0, w, h));
img1.copyTo(half);
"""

"""https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html"""

"""https://docs.opencv.org/3.4.1/d9/dab/tutorial_homography.html"""

"""https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html"""

"""https://docs.opencv.org/3.1.0/da/d6e/tutorial_py_geometric_transformations.html"""

"""3https://docs.opencv.org/3.4.1/d9/dab/tutorial_homography.html"""

"""https://github.com/tsherlock/panorama/blob/master/my_panos/pano_stitcher.py"""
