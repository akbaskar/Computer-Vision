import cv2
import numpy as np
import math
img = cv2.imread("task2.jpg",0)
#img1 = cv2.imread("task2.jpg")
def convertTolist(img1):
    rowLen = len(img1)
    colLen = len(img1[0])
    outputMat = [[0 for col in range(colLen)] for row in range(rowLen)]
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            outputMat[i][j] = img1[i][j]
    return outputMat

originalimg = convertTolist(img)

def resize(img, i):
    a = img[::i]
    b = []
    for col in a:
        b.append( col[::i])
    return b


def absoluteVal(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):

            if matrix[i][j]  < 0 :
                matrix[i][j] *= -1
    return matrix

def maxVal(matrix):
    maxVal = matrix[0][0]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]  > maxVal :
                maxVal = matrix[i][j]
    return maxVal

#pos_halfImg = np.asarray(absoluteVal(halfImg)) / maxVal(absoluteVal(halfImg))
oct1 = resize(img , 1)
oct2 = resize(img , 2)
oct3 = resize(img , 4)
oct4 = resize(img , 8)

print("oct1 resolution:")
print(len(oct1))
print(len(oct1[0]))
print("oct2 resolution:")
print(len(oct2))
print(len(oct2[0]))
print("oct3 resolution:")
print(len(oct3))
print(len(oct3[0]))
print("oct4 resolution:")
print(len(oct4))
print(len(oct4[0]))


a = absoluteVal(oct1)
oct1 = a/maxVal(a)
a = absoluteVal(oct2)
oct2 = a/maxVal(a)
a = absoluteVal(oct3)
oct3 = a/maxVal(a)
a = absoluteVal(oct4)
oct4 = a/maxVal(a)

"""
a = np.asarray(oct1)
cv2.imwrite("resize by 0_abs.jpg",a)
a = np.asarray(oct2)
cv2.imwrite("resize by 2_abs.jpg",a)
a = np.asarray(oct3)
cv2.imwrite("resize by 4_abs.jpg",a)
a = np.asarray(oct4)
cv2.imwrite("resize by 8_abs.jpg",a)
"""

#taskimg = convertTolist(img1)
oct1 = convertTolist(oct1)
oct2 = convertTolist(oct2)
oct3 = convertTolist(oct3)
oct4 = convertTolist(oct4)

orgoct1 = convertTolist(oct1)
orgoct2 = convertTolist(oct2)
orgoct3 = convertTolist(oct3)
orgoct4 = convertTolist(oct4)

def gfn(x,y,sigma):
    a = 1/(2 * math.pi * sigma*sigma)
    b = (-1) * ((x*x) + (y*y)) / (2 * sigma * sigma)
    return a * math.exp(b)

def findc(gmat):
    sum = 0
    for i in gmat:
        for j in i:
            sum = sum + j
    return 1/sum

def mulc(gmat):
    c = findc(gmat)
    result = []
    for i in gmat:
        result.append([j * c for j in i])
    return result

sigma1 = [1/math.sqrt(2),1,math.sqrt(2),2,2* math.sqrt(2)]
sigma2 = [math.sqrt(2),2,2* math.sqrt(2),4,4*math.sqrt(2)]
sigma3 = [2*math.sqrt(2),4,4*math.sqrt(2),8,8*math.sqrt(2)]
sigma4 = [4*math.sqrt(2),8,8*math.sqrt(2),16,16*math.sqrt(2)]

def creategmat(sigma):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []

    for i in range(-3,4):
        a = str(i) + ",3"
        list1.append(gfn(i,3,sigma))
        a = str(i) + ",2"
        list2.append(gfn(i,2,sigma))
        a = str(i) + ",1"
        list3.append(gfn(i,1,sigma))
        a = str(i) + ",0"
        list4.append(gfn(i,0,sigma))
        a = str(i) + ",-1"
        list5.append(gfn(i,-1,sigma))
        a = str(i) + ",-2"
        list6.append(gfn(i,-2,sigma))
        a = str(i) + ",-3"
        list7.append(gfn(i,-3,sigma))

    gmat = [list1,list2,list3,list4,list5,list6,list7]
    gmat = mulc(gmat)
    return gmat

#Gaussian matrixes for first octave
gmat11 = creategmat(sigma1[0])
gmat12 = creategmat(sigma1[1])
gmat13 = creategmat(sigma1[2])
gmat14 = creategmat(sigma1[3])
gmat15 = creategmat(sigma1[4])

#Gaussian matrixes for second octave
gmat21 = creategmat(sigma2[0])
gmat22 = creategmat(sigma2[1])
gmat23 = creategmat(sigma2[2])
gmat24 = creategmat(sigma2[3])
gmat25 = creategmat(sigma2[4])

#Gaussian matrixes for third octave
gmat31 = creategmat(sigma3[0])
gmat32 = creategmat(sigma3[1])
gmat33 = creategmat(sigma3[2])
gmat34 = creategmat(sigma3[3])
gmat35 = creategmat(sigma3[4])

#Gaussian matrixes for forth octave
gmat41 = creategmat(sigma4[0])
gmat42 = creategmat(sigma4[1])
gmat43 = creategmat(sigma4[2])
gmat44 = creategmat(sigma4[3])
gmat45 = creategmat(sigma4[4])

def zeropad(img):
    rowLen = len(img)
    colLen = len(img[0])
    for i in img:
        i.insert(0,0)
        i.append(0)
    img.insert(0,[0 for col in range(colLen+2)])
    img.append([0 for col in range(colLen+2)])
    return img

def pad7(paddedimg,rowLim,colLim):

    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []

    for c in colLim:
        list1.append(paddedimg[rowLim[0]][c])
        list2.append(paddedimg[rowLim[1]][c])
        list3.append(paddedimg[rowLim[2]][c])
        list4.append(paddedimg[rowLim[3]][c])
        list5.append(paddedimg[rowLim[4]][c])
        list6.append(paddedimg[rowLim[5]][c])
        list7.append(paddedimg[rowLim[6]][c])
    paddedimg7c7 = [list1,list2,list3,list4,list5,list6,list7]
    return paddedimg7c7

def dotProd(paddedimg, img2 ,row, col):
    rowLim = [row-3,row-2,row-1,row,row+1,row+2,row+3]
    colLim = [col-3,col-2,col-1,col,col+1,col+2,col+3]
    sum = 0
    paddedimg7c7 = pad7(paddedimg,rowLim,colLim)
    for i in range(7):
        for j in range(7):
            sum = sum + (paddedimg7c7[i][j] * img2[i][j])
    return sum

def convolution(img1, img2):
    rowLen = len(img1)
    colLen = len(img1[0])
    paddedimg = zeropad(zeropad(zeropad(img1)))
    outputMat = [[0 for col in range(colLen)] for row in range(rowLen)]
    for i in range(rowLen):
        for j in range(colLen):
            outputMat[i][j] = dotProd(paddedimg, img2, i+3,j+3)
    return outputMat


"""
gmat11 = np.asarray(gmat11)
gmat12 = np.asarray(gmat12)
gmat13 = np.asarray(gmat13)
gmat14 = np.asarray(gmat14)
gmat15 = np.asarray(gmat15)
"""
#img = cv2.imread("task2.jpg",0)
oct1 = convertTolist(orgoct1)
convOct11 = convolution(oct1,gmat11)
a = absoluteVal(convOct11)
convOct11 = a/maxVal(a)
#convOct11 = cv2.filter2D(img,-1, gmat11)
oct1 = convertTolist(orgoct1)
convOct12 = convolution(oct1,gmat12)
a = absoluteVal(convOct12)
convOct12 = a/maxVal(a)
#convOct12 = cv2.filter2D(img,-1, gmat12)
oct1 = convertTolist(orgoct1)
convOct13 = convolution(oct1,gmat13)
a = absoluteVal(convOct13)
convOct13 = a/maxVal(a)
#convOct13 = cv2.filter2D(img,-1, gmat13)
oct1 = convertTolist(orgoct1)
convOct14 = convolution(oct1,gmat14)
a = absoluteVal(convOct14)
convOct14 = a/maxVal(a)
#convOct14 = cv2.filter2D(img,-1, gmat14)
oct1 = convertTolist(orgoct1)
convOct15 = convolution(oct1,gmat15)
a = absoluteVal(convOct15)
convOct15 = a/maxVal(a)
#convOct15 = cv2.filter2D(img,-1, gmat15)
oct1 = convertTolist(orgoct1)

"""
print("lenth of convOct11")
print(len(convOct11))
print(len(convOct11[0]))

print("lenth of convOct12")
print(len(convOct12))
print(len(convOct12[0]))

print("lenth of convOct13")
print(len(convOct13))
print(len(convOct13[0]))

print("lenth of convOct14")
print(len(convOct14))
print(len(convOct14[0]))

print("lenth of convOct15")
print(len(convOct15))
print(len(convOct15[0]))
"""

oct2 = convertTolist(orgoct2)
convOct21 = convolution(oct2,gmat21)

oct2 = convertTolist(orgoct2)
convOct22 = convolution(oct2,gmat22)

oct2 = convertTolist(orgoct2)
convOct23 = convolution(oct2,gmat23)

oct2 = convertTolist(orgoct2)
convOct24 = convolution(oct2,gmat24)
oct2 = convertTolist(orgoct2)

convOct25 = convolution(oct2,gmat25)

"""
o21 = np.asarray(convOct21)
cv2.imwrite("convOct21.jpg",o21)
o22 = np.asarray(convOct22)
cv2.imwrite("convOct22.jpg",o22)
o23 = np.asarray(convOct23)
cv2.imwrite("convOct23.jpg",o23)
o24 = np.asarray(convOct24)
cv2.imwrite("convOct24.jpg",o24)
o25 = np.asarray(convOct25)
cv2.imwrite("convOct25.jpg",o25)
"""

a = absoluteVal(convOct21)
convOct21 = a/maxVal(a)
a = absoluteVal(convOct22)
convOct22 = a/maxVal(a)
a = absoluteVal(convOct23)
convOct23 = a/maxVal(a)
a = absoluteVal(convOct24)
convOct24 = a/maxVal(a)
a = absoluteVal(convOct25)
convOct25 = a/maxVal(a)
"""
o11 = np.asarray(convOct11)
cv2.imwrite("convOct11.jpg",o11)
cv2.imshow("Oct11",a)
cv2.waitKey(0)
cv2.destroyAllWindows()
o12 = np.asarray(convOct12)
cv2.imwrite("convOct12.jpg",o12)
cv2.imshow("Oct12",a)
cv2.waitKey(0)
cv2.destroyAllWindows()
o13 = np.asarray(convOct13)
cv2.imwrite("convOct13.jpg",o13)
cv2.imshow("Oct13",a)
cv2.waitKey(0)
cv2.destroyAllWindows()
o14 = np.asarray(convOct14)
cv2.imwrite("convOct14.jpg",o14)
cv2.imshow("Oct14",a)
cv2.waitKey(0)
cv2.destroyAllWindows()
o15 = np.asarray(convOct15)
cv2.imwrite("convOct15.jpg",o15)
cv2.imshow("Oct15",a)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


oct3 = convertTolist(orgoct3)
convOct31 = convolution(oct3,gmat31)
oct3 = convertTolist(orgoct3)
convOct32 = convolution(oct3,gmat32)
oct3 = convertTolist(orgoct3)
convOct33 = convolution(oct3,gmat33)
oct3 = convertTolist(orgoct3)
convOct34 = convolution(oct3,gmat34)
oct3 = convertTolist(orgoct3)
convOct35 = convolution(oct3,gmat35)

"""
a = np.asarray(convOct31)
cv2.imwrite("convOct31.jpg",a)
a = np.asarray(convOct32)
cv2.imwrite("convOct32.jpg",a)
a = np.asarray(convOct33)
cv2.imwrite("convOct33.jpg",a)
a = np.asarray(convOct34)
cv2.imwrite("convOct34.jpg",a)
a = np.asarray(convOct35)
cv2.imwrite("convOct35.jpg",a)
"""

a = absoluteVal(convOct31)
convOct31 = a/maxVal(a)
a = absoluteVal(convOct32)
convOct32 = a/maxVal(a)
a = absoluteVal(convOct33)
convOct33 = a/maxVal(a)
a = absoluteVal(convOct34)
convOct34 = a/maxVal(a)
a = absoluteVal(convOct35)
convOct35 = a/maxVal(a)




oct4 = convertTolist(orgoct4)
convOct41 = convolution(oct4,gmat41)
a = absoluteVal(convOct41)
convOct41 = a/maxVal(a)
oct4 = convertTolist(orgoct4)
convOct42 = convolution(oct4,gmat42)
a = absoluteVal(convOct42)
convOct42 = a/maxVal(a)
oct4 = convertTolist(orgoct4)
convOct43 = convolution(oct4,gmat43)
a = absoluteVal(convOct43)
convOct43 = a/maxVal(a)
oct4 = convertTolist(orgoct4)
convOct44 = convolution(oct4,gmat44)
a = absoluteVal(convOct44)
convOct44 = a/maxVal(a)
oct4 = convertTolist(orgoct4)
convOct45 = convolution(oct4,gmat45)
a = absoluteVal(convOct45)
convOct45 = a/maxVal(a)


def DoG(mat1, mat2):
    rowLen = len(mat1)
    colLen = len(mat1[0])
    outputMat = [[0 for col in range(colLen)] for row in range(rowLen)]
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            outputMat[i][j] = mat1[i][j] - mat2[i][j]
    return outputMat

#Difference of Gaussing Matrix in octave 1
dog11 = DoG(convOct11,convOct12)
dog12 = DoG(convOct12,convOct13)
dog13 = DoG(convOct13,convOct14)
dog14 = DoG(convOct14,convOct15)

a = np.asarray(dog11)
cv2.imwrite("DoG11.jpg",a)
a = np.asarray(dog12)
cv2.imwrite("DoG12.jpg",a)
a = np.asarray(dog13)
cv2.imwrite("DoG13.jpg",a)
a = np.asarray(dog14)
cv2.imwrite("DoG14.jpg",a)

"""
print("length of dog11")
print(len(dog11))
print(len(dog11[0]))

print("length of dog12")
print(len(dog12))
print(len(dog12[0]))

print("length of dog13")
print(len(dog13))
print(len(dog13[0]))

print("length of dog14")
print(len(dog14))
print(len(dog14[0]))
"""

#Difference of Gaussing Matrix in octave 2
dog21 = DoG(convOct21,convOct22)
dog22 = DoG(convOct22,convOct23)
dog23 = DoG(convOct23,convOct24)
dog24 = DoG(convOct24,convOct25)

a = np.asarray(dog21)
cv2.imwrite("DoG21.jpg",a)
a = np.asarray(dog22)
cv2.imwrite("DoG22.jpg",a)
a = np.asarray(dog23)
cv2.imwrite("DoG23.jpg",a)
a = np.asarray(dog24)
cv2.imwrite("DoG24.jpg",a)


#Difference of Gaussing Matrix in octave 3
dog31 = DoG(convOct31,convOct32)
dog32 = DoG(convOct32,convOct33)
dog33 = DoG(convOct33,convOct34)
dog34 = DoG(convOct34,convOct35)

a = np.asarray(dog31)
cv2.imwrite("DoG31.jpg",a)
a = np.asarray(dog32)
cv2.imwrite("DoG32.jpg",a)
a = np.asarray(dog33)
cv2.imwrite("DoG33.jpg",a)
a = np.asarray(dog34)
cv2.imwrite("DoG34.jpg",a)

#Difference of Gaussing Matrix in octave 4
dog41 = DoG(convOct41,convOct42)
dog42 = DoG(convOct42,convOct43)
dog43 = DoG(convOct43,convOct44)
dog44 = DoG(convOct44,convOct45)


def findMinIn3c3(mat3c3):
    list = []
    for i in mat3c3:
        for j in i:
            list.append(j)
    return min(list)

def findMaxIn3c3(mat3c3):
    list = []
    for i in mat3c3:
        for j in i:
            list.append(j)
    return max(list)

def findMinIn3c3WithoutCenterElement(mat3c3):
    list = []
    for i in range(3):
        for j in range(3):
            list.append(mat3c3[i][j])
    list.pop(4)
    return min(list)

def findMaxIn3c3WithoutCenterElement(mat3c3):
    list = []
    for i in range(3):
        for j in range(3):
                list.append(mat3c3[i][j])
    list.pop(4)
    return max(list)

def getCenterElement(mat3c3):
    list = []
    for i in range(3):
        for j in range(3):
                list.append(mat3c3[i][j])
    return list.pop(4)

def get3c3Mat(mat,row,col):
    rowLim = [row-1,row,row+1]
    colLim = [col-1,col,col+1]
    list1 = []
    list2 = []
    list3 = []
    for c in colLim:
        list1.append(mat[rowLim[0]][c])
        list2.append(mat[rowLim[1]][c])
        list3.append(mat[rowLim[2]][c])
    mat3c3 = [list1,list2,list3]
    return mat3c3

def markKeyPoint(img,x,y,i):
    img[x*i][y*i] = 255
    return img

#oct1WithKeyPoints1 = convertTolist(orgoct1)
#oct1WithKeyPoints2 = convertTolist(orgoct1)
#oct2WithKeyPoints1 = convertTolist(orgoct2)
#oct2WithKeyPoints2 = convertTolist(orgoct2)


def findKeyPoints(dog11,dog12,dog13,resizeFactor, str):
    img1 = cv2.imread("task2.jpg")
    for i in range(1,len(dog11)-1):
        for j in range(1,len(dog11[0])-1):
            mat3c31 = get3c3Mat(dog11,i,j)
            min1 = findMinIn3c3(mat3c31)
            max1 = findMaxIn3c3(mat3c31)

            mat3c32 = get3c3Mat(dog12,i,j)
            point = getCenterElement(mat3c32)
            min2 = findMinIn3c3WithoutCenterElement(mat3c32)
            max2 = findMaxIn3c3WithoutCenterElement(mat3c32)

            mat3c33 = get3c3Mat(dog13,i,j)
            min3 = findMinIn3c3(mat3c33)
            max3 = findMaxIn3c3(mat3c33)

            overallMin = min(min1,min2,min3)
            overallMax = max(max1,max2,max3)

            if(point < overallMin or point > overallMax):
                #taskimg = markKeyPoint(taskimg,i,j,resizeFactor)
                img1[i*resizeFactor][j*resizeFactor] = 255
                img2[i*resizeFactor][j*resizeFactor] = 255
    cv2.imwrite(str + ".jpg",img1)

img2 = cv2.imread("task2.jpg")
findKeyPoints(dog11,dog12,dog13,1,"oct1WithKeyPoints1")
findKeyPoints(dog12,dog13,dog14,1,"oct1WithKeyPoints2")
findKeyPoints(dog21,dog22,dog23,1,"oct2WithKeyPoints1")
findKeyPoints(dog22,dog23,dog24,1,"oct2WithKeyPoints2")
findKeyPoints(dog31,dog32,dog33,1,"oct3WithKeyPoints1")
findKeyPoints(dog32,dog33,dog34,1,"oct3WithKeyPoints2")
findKeyPoints(dog41,dog42,dog43,1,"oct4WithKeyPoints1")
findKeyPoints(dog42,dog43,dog44,1,"oct4WithKeyPoints2")

cv2.imwrite("FinalWithAllKeyPoints.jpg",img2)
#taskimg = convertTolist(img1)
#print(type(taskimg))
#print(type(img1))
#oct1WithKeyPoints1 = findKeyPoints(dog11,dog12,dog13,1)
#taskimg = convertTolist(img1)
#oct1WithKeyPoints2 = findKeyPoints(dog12,dog13,dog14,1)

#taskimg = convertTolist(img1)
#oct2WithKeyPoints1 = findKeyPoints(dog21,dog22,dog23,2)
#taskimg = convertTolist(img1)
#oct2WithKeyPoints2 = findKeyPoints(dog22,dog23,dog24,2)

#print(type(oct1WithKeyPoints1))

"""
o11 = np.asarray(oct1WithKeyPoints1,dtype = 'float32')
print(type(o11))
print(type(o11[0]))
cv2.imwrite("oct1WithKeyPoints1.jpg",o11)
cv2.imshow("oct1WithKeyPoints1",o11)
cv2.waitKey(0)
cv2.destroyAllWindows()

o11 = np.asarray(oct1WithKeyPoints1)
cv2.imwrite("oct1WithKeyPoints2.jpg",o11)

o11 = np.asarray(oct2WithKeyPoints1)
cv2.imwrite("oct2WithKeyPoints1.jpg",o11)

o11 = np.asarray(oct2WithKeyPoints2)
cv2.imwrite("oct2WithKeyPoints2.jpg",o11)
"""
