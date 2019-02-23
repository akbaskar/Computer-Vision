import cv2
import numpy as np

img = cv2.imread("noise.jpg",0)
# Image shape 310 x 351

def zeropad(img):
    img1 = np.insert(img,0,0,axis=1)
    img1 = np.insert(img1,len(img[0])+1,0,axis=1)
    img1 = np.insert(img1,0,0,axis=0)
    img1 = np.insert(img1,len(img)+1,0,axis=0)
    return img1

def submatrix(matrix, startRow, startCol, size):
    return matrix[startRow-1:startRow-1+size,startCol-1:startCol-1+size]

def dilation(img):
    rowLen = len(img)
    colLen = len(img[0])
    outputMat = [[0 for col in range(colLen)] for row in range(rowLen)]
    outputMat = np.asarray(outputMat)
    img1 = zeropad (img)

    for r in range(1,len(img)-1):
        for c in range(1,len(img[0])-1):
            outputMat[r-1][c-1] = submatrix(img1,r,c,3).max()
    return outputMat

def erosion(img):
    rowLen = len(img)
    colLen = len(img[0])
    outputMat = [[0 for col in range(colLen)] for row in range(rowLen)]
    outputMat = np.asarray(outputMat)
    img1 = zeropad (img)

    for r in range(1,len(img)-1):
        for c in range(1,len(img[0])-1):
            outputMat[r-1][c-1] = submatrix(img1,r,c,3).min()
    return outputMat

def closing(img):
    return erosion(dilation(img))

def opening(img):
    return dilation(erosion(img))

a= opening(img)
cv2.imwrite("opening.jpg",a)

b= closing(img)
cv2.imwrite("closing.jpg",b)

res1 = opening(closing(img))
cv2.imwrite("res_noise1.jpg",res1)

res2 = closing(opening(img))
cv2.imwrite("res_noise2.jpg",res2)

# Boundary = A - A erosion B
def findBoundary(img):
    erodedimg = erosion(img)
    return img - erodedimg

res_bound1 = findBoundary(res1)
cv2.imwrite("res_bound1.jpg",res_bound1)

res_bound2 = findBoundary(res2)
cv2.imwrite("res_bound2.jpg",res_bound2)

diff = res1 - res2
cv2.imwrite("difference.jpg",diff)
