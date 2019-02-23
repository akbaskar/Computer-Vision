import cv2
import numpy as np
import math

img = cv2.imread("hough.jpg",0)
img[:,0:2] = img[:,3:4]
img_output = cv2.imread("hough.jpg")
#img_output[:,0:2] = img_output[:,3:4]
#edges = cv2.Canny(img,100,200)

im_gray = cv2.imread('hough.jpg', 0)
img = cv2.threshold(im_gray, 112, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('binary_image.png', img)

sobelx = [[-1,0,1],[-2,0,2],[-1,0,1]]
sobely = [[1,2,1],[0,0,0],[-1,-2,-1]]

def zeropad(img):
    img1 = np.insert(img,0,0,axis=1)
    img1 = np.insert(img1,len(img[0])+1,0,axis=1)
    img1 = np.insert(img1,0,0,axis=0)
    img1 = np.insert(img1,len(img)+1,0,axis=0)
    return img1

def convertTolist(img1):
    rowLen = len(img1)
    colLen = len(img1[0])
    outputMat = [[0 for col in range(colLen)] for row in range(rowLen)]
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            outputMat[i][j] = img1[i][j]
    return outputMat

rowLen,colLen = img.shape
img1 = convertTolist(img)
#img1 = np.asarray(img)
paddedimg = zeropad(img1)
outputMatx = [[0 for col in range(colLen)] for row in range(rowLen)]
outputMaty = [[0 for col in range(colLen)] for row in range(rowLen)]

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

for i in range(rowLen):
    for j in range(colLen):
        outputMatx[i][j] = dotProdx(i+1,j+1)

for i in range(rowLen):
    for j in range(colLen):
        #outputMat[i][j]
        outputMaty[i][j] = dotPrody(i+1,j+1)

sobelX = np.uint8(np.abs(outputMatx))
sobelY = np.uint8(np.abs(outputMaty))
sobelCombined = cv2.bitwise_or(sobelX, sobelY)
edges1 = np.asarray(sobelCombined)
cv2.imwrite("sobel.jpg",edges1)

height, width = img.shape # we need heigth and width to calculate the diag
img_diagonal = np.ceil(np.sqrt(height**2 + width**2))
x_arr,y_arr = np.nonzero(edges1)
thetas = np.deg2rad(np.arange(-90, -80, 1))
# thetas = np.deg2rad(np.arange(-60, -50, 1))

def get_accumulator_matrix(img,edges,thetas):
    height, width = img.shape
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2))
    rhos = np.arange(-img_diagonal, img_diagonal + 1, 1)
    #thetas = np.deg2rad(np.arange(-90, -80, 1))
    #thetas = np.deg2rad(np.arange(-60, -50, 1))
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_i, x_i = np.nonzero(edges)
    for i in range(len(x_i)):
        x = x_i[i]
        y = y_i[i]
        for j in range(len(thetas)):
            rho = int((y * np.cos(thetas[j]) + x * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1
    return H

def optimize_peak(img,n_iter):
    img1 = zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(zeropad(img)))))))))))))))))))))))))
    x_list =[]
    y_list = []
    for i in range(n_iter):
        max_i = np.argmax(img1) # find argmax in flattened array
        x_i,y_i = np.unravel_index(max_i, img1.shape) # remap to shape of H
        x_list.append(x_i-25)
        y_list.append(y_i-25)
        for i in range(x_i-25,x_i+25):
            for j in range(y_i-25,y_i+25):
                    img1[i][j] = 0
    return x_list,y_list


def findredlines(img,edges):
    thetas = np.deg2rad(np.arange(-90, -80, 1))
    accumulator = get_accumulator_matrix(img,edges,thetas)
    peak_rhos,peak_thetas1 = optimize_peak(accumulator,8)
    peak_thetas = []
    for i in peak_thetas1:
        peak_thetas.append(thetas[i])
    for i in range (len(peak_thetas)):
        y1= (peak_rhos[i]-img_diagonal - 0 * math.cos(peak_thetas[i]))/math.sin(peak_thetas[i])
        y2= (peak_rhos[i]-img_diagonal - 1000 * math.cos(peak_thetas[i]))/math.sin(peak_thetas[i])
        cv2.line(img_output,(int(y1),0),(int(y2),1000),(0,0,255),2)

    cv2.imwrite("red_lines.jpg",img_output)

def findbluelines(img,edges):
    thetas = np.deg2rad(np.arange(-60, -50, 1))
    accumulator = get_accumulator_matrix(img,edges,thetas)
    peak_rhos,peak_thetas1 = optimize_peak(accumulator,7)
    peak_thetas = []
    for i in peak_thetas1:
        peak_thetas.append(thetas[i])

    img_output = cv2.imread("hough.jpg")
    for i in range (len(peak_thetas)):
        y1= (peak_rhos[i]-img_diagonal - 0 * math.cos(peak_thetas[i]))/math.sin(peak_thetas[i])
        y2= (peak_rhos[i]-img_diagonal - 1000 * math.cos(peak_thetas[i]))/math.sin(peak_thetas[i])
        cv2.line(img_output,(int(y1),0),(int(y2),1000),(255,0,0),2)

    cv2.imwrite("blue_lines.jpg",img_output)

findredlines(img,edges1)
findbluelines(img,edges1)

"""
print("------")
print(peak_thetas)
print(peak_rhos)
print(len(peak_thetas))
print(len(peak_rhos))

for i in range (len(peak_thetas)):
    y1= (peak_rhos[i]-img_diagonal - 0 * math.cos(peak_thetas[i]))/math.sin(peak_thetas[i])
    y2= (peak_rhos[i]-img_diagonal - 1000 * math.cos(peak_thetas[i]))/math.sin(peak_thetas[i])
    cv2.line(img_output,(int(y1),0),(int(y2),1000),(0,0,255),2)

cv2.imwrite("red_lines.jpg",img_output)
"""

a = x * cos theta
b = y * sin theta

def get_accumulator_matrix_circle(img,edges,thetas):
    height, width = img.shape
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2))
    rhos = np.arange(-img_diagonal, img_diagonal + 1, 1)
    #thetas = np.deg2rad(np.arange(-90, -80, 1))
    #thetas = np.deg2rad(np.arange(-60, -50, 1))
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_i, x_i = np.nonzero(edges)
    for i in range(len(x_i)):
        x = x_i[i]
        y = y_i[i]
        for j in range(len(thetas)):
            rho = int((y * np.cos(thetas[j]) + x * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1
    return H

def findcircles(img,edges):
    thetas = np.deg2rad(np.arange(0, 360, 1))
    accumulator = get_accumulator_matrix(img,edges,thetas)
    peak_rhos,peak_thetas1 = optimize_peak(accumulator,7)
    peak_thetas = []
    for i in peak_thetas1:
        peak_thetas.append(thetas[i])

    img_output = cv2.imread("hough.jpg")
    for i in range (len(peak_thetas)):
        y1= (peak_rhos[i]-img_diagonal - 0 * math.cos(peak_thetas[i]))/math.sin(peak_thetas[i])
        y2= (peak_rhos[i]-img_diagonal - 1000 * math.cos(peak_thetas[i]))/math.sin(peak_thetas[i])
        cv2.line(img_output,(int(y1),0),(int(y2),1000),(255,0,0),2)

    cv2.imwrite("blue_lines.jpg",img_output)
