import cv2
import numpy as np

img = cv2.imread("task1.png",0)

sobelx = [[-1,0,1],[-2,0,2],[-1,0,1]]
sobely = [[1,2,1],[0,0,0],[-1,-2,-1]]

#paddedimg = np.pad(img, pad_width=1, mode='constant', constant_values=0)
def zeropad(img):
    rowLen = len(img)
    colLen = len(img[0])
    for i in img:
        i.insert(0,0)
        i.append(0)
    img.insert(0,[0 for col in range(colLen+2)])
    img.append([0 for col in range(colLen+2)])
    return img

def convertTolist(img1):
    rowLen = len(img1)
    colLen = len(img1[0])
    outputMat = [[0 for col in range(colLen)] for row in range(rowLen)]
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            outputMat[i][j] = img1[i][j]
    return outputMat

img1 = convertTolist(img)
paddedimg = zeropad(img1)
outputMat = [[0 for col in range(900)] for row in range(600)]

def pad3(rowLim, colLim):
    #paddedimg3 = [[0 for col in range(3)] for row in range(3)]
    list1 = []
    list2 = []
    list3 = []
    for c in colLim:
        list1.append(paddedimg[rowLim[0]][c])
        list2.append(paddedimg[rowLim[1]][c])
        list3.append(paddedimg[rowLim[2]][c])
    paddedimg3 = [list1,list2,list3]
    return paddedimg3

def dotProdx(row, col):
    rowLim = [row-1,row,row+1]
    colLim = [col-1,col,col+1]
    sum = 0
    paddedimg3 = pad3(rowLim,colLim)
    for i in range(3):
        for j in range(3):
            #print("sobel =", sobelx[i][j],"paddedimg =  ", paddedimg3[i][j])
            sum = sum + (sobelx[i][j] * paddedimg3[i][j])
    return sum
#outputMat[0][0] = dotProd(1,1)

def dotPrody(row, col):
    rowLim = [row-1,row,row+1]
    colLim = [col-1,col,col+1]
    sum = 0
    paddedimg3 = pad3(rowLim,colLim)
    for i in range(3):
        for j in range(3):
            #print("sobel =", sobelx[i][j],"paddedimg =  ", paddedimg3[i][j])
            sum = sum + (sobely[i][j] * paddedimg3[i][j])
    return sum

for i in range(600):
    for j in range(900):
        outputMat[i][j] = dotProdx(i+1,j+1)


a = np.asarray(outputMat)
cv2.imwrite("Sobel_for_x.jpg", a)
cv2.imshow("output image for x", a)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(600):
    for j in range(900):
        outputMat[i][j] = dotPrody(i+1,j+1)

a = np.asarray(outputMat)
cv2.imwrite("Sobel_for_y.jpg", a)

#def padding(image):
"""cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
