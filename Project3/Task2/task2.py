import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("point.jpg",0)

ker = [[-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, 24, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]]
ker = np.asarray(ker)

def zeropad(img):
    img1 = np.insert(img,0,0,axis=1)
    img1 = np.insert(img1,len(img[0])+1,0,axis=1)
    img1 = np.insert(img1,0,0,axis=0)
    img1 = np.insert(img1,len(img)+1,0,axis=0)
    return img1

def pad5(paddedimg,rowLim,colLim):

    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []

    for c in colLim:
        list1.append(paddedimg[rowLim[0]][c])
        list2.append(paddedimg[rowLim[1]][c])
        list3.append(paddedimg[rowLim[2]][c])
        list4.append(paddedimg[rowLim[3]][c])
        list5.append(paddedimg[rowLim[4]][c])
        #list6.append(paddedimg[rowLim[5]][c])
        #list7.append(paddedimg[rowLim[6]][c])
    paddedimg5c5 = [list1,list2,list3,list4,list5]
    return paddedimg5c5

def submatrix(matrix, startRow, startCol, size):
    return matrix[startRow-1:startRow-1+size,startCol-1:startCol-1+size]

def dotProd(paddedimg, img2 ,row, col):
    rowLim = [row-2,row-1,row,row+1,row+2]
    colLim = [col-2,col-1,col,col+1,col+2]
    sum = 0
    paddedimg5c5 = pad5(paddedimg,rowLim,colLim)
    for i in range(5):
        for j in range(5):
            sum = sum + (paddedimg5c5[i][j] * img2[i][j])
    return sum

def convolution(img1, img2):
    rowLen = len(img1)
    colLen = len(img1[0])
    paddedimg = zeropad(zeropad(img1))
    #outputMat = np.zeros(rowLen,colLen)
    outputMat = [[0 for col in range(colLen)] for row in range(rowLen)]
    for i in range(rowLen):
        for j in range(colLen):
            outputMat[i][j] = dotProd(paddedimg, img2, i+2,j+2)
    return outputMat

output = convolution(img,ker)
output = np.asarray(output)
cv2.imwrite("output before thresholding.jpg",output)

def get_threshold(img):
    max = np.max(img)
    return max*0.9

output = np.absolute(output)
max = get_threshold(output)
x,y = np.nonzero(output>max)
print(np.nonzero(output>max)) # 249,445
output[output < max] = 0
output[output > max] = 255

cv2.imwrite("output.jpg",output)


print(x)
print(y)

"""
for i in range(len(x)):
    x1 = x[i]
    y1 = y[i]
    cv2.circle(output, (x,y),10, (255,255,255), thickness=2, lineType=8, shift=0)
#cv2.circle(output, (x[0],y[0]),10, (255,255,255), thickness=2, lineType=8, shift=0)
cv2.imwrite("output1.jpg",output)

"""

# Part 2b
img = cv2.imread("segment.jpg",0)

H = np.zeros(256)
row,col = img.shape

for i in range(row):
    for j in range(col):
        H[img[i][j]] += 1

H[0] = 0
plt.bar(range(256),(H))
#plt.plot(H)
plt.show()

threshold = 205

for i in range(row):
    for j in range(col):
        if img[i][j] < threshold:
            img[i][j] = 0
        else:
            img[i][j] = 255

cv2.imwrite("segmented_img.jpg",img)
