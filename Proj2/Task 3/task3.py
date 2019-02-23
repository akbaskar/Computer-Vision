import numpy as np
import math
import cv2
import random
import matplotlib.pyplot as plt
UBIT = "BASKARAD"
np.random.seed(sum([ord(c) for c in UBIT]))


x = [[5.9, 3.2],
     [4.6, 2.9],
     [6.2, 2.8],
     [4.7, 3.2],
     [5.5, 4.2],
     [5.0, 3.0],
     [4.9, 3.1],
     [6.7, 3.1],
     [5.1, 3.8],
     [6.0, 3.0]]

x = np.asarray(x)

def dist(p1, p2):
    return math.sqrt(math.pow((p1[0] - p2[0]),2) + math.pow((p1[1] - p2[1]),2))

mu = [[6.2, 3.2], [6.6, 3.7], [6.2, 3.0]]

def classify(x,mu):
    dict = {}
    for j in range(len(mu)):
        dict[j] = []

    for xy in x:
        """d1 = dist(m1,xy)
        d2 = dist(m2,xy)
        d3 = dist(m3,xy)
        d = [d1,d2,d3]"""
        d = []
        for m in mu:
            d.append(dist(m,xy))
        index = d.index(min(d))
        dict[index].append(xy)

    for j in range(len(mu)):
        dict[j] = np.asarray(dict[j])
    return dict

def plotpoints(dict,count):
    plt.scatter(dict[0][:,0],dict[0][:,1],marker='^',c='r',edgecolors='face')
    plt.scatter(dict[1][:,0],dict[1][:,1],marker='^',c='g',edgecolors='face')
    plt.scatter(dict[2][:,0],dict[2][:,1],marker='^',c='b',edgecolors='face')

    for i in range(len(dict)):
        for j in range(len(dict[i])):
            plt.text(dict[i][j][0],dict[i][j][1],'   ' + '(' + str(dict[i][j][0]) + ' , ' + str(dict[i][j][1]) + ')')

    plt.savefig('task3_iter' + str(count+1) + '_a.png')
    #plt.imsave('task3_iter' + str(count+1) + '_a', img)
    #plt.show()

def plotpointsWithM(dict,mu,count):
    plt.scatter(dict[0][:,0],dict[0][:,1],marker='^',c='r',edgecolors='face')
    plt.scatter(dict[1][:,0],dict[1][:,1],marker='^',c='g',edgecolors='face')
    plt.scatter(dict[2][:,0],dict[2][:,1],marker='^',c='b',edgecolors='face')

    for i in range(len(dict)):
        for j in range(len(dict[i])):
            plt.text(dict[i][j][0],dict[i][j][1],'   ' + '(' + str(dict[i][j][0]) + ' , ' + str(dict[i][j][1]) + ')')

    plt.scatter(mu[0][0],mu[0][1],marker='o',c='r',edgecolors='face')
    plt.scatter(mu[1][0],mu[1][1],marker='o',c='g',edgecolors='face')
    plt.scatter(mu[2][0],mu[2][1],marker='o',c='b',edgecolors='face')

    for i in range(len(mu)):
        plt.text(mu[i][0],mu[i][1],'   ' + '(' + str(mu[i][0]) + ' , ' + str(mu[i][1]) + ')')

    plt.savefig('task3_iter' + str(count+1) + '_b.png')
    #plt.show()

def updateMean(dict,mu):
    #updating the Mu1,2,3 by finding the mean
    for i in range(len(mu)):
        mu[i][0],mu[i][1] = np.mean(dict[i][:,0]),np.mean(dict[i][:,1])
    #print("Mean values updated. Updated values are: ")
    #print(mu)
    return mu
    """m1[0],m1[1] = np.mean(dict[0][:,0]),np.mean(dict[0][:,1])
    m2[0],m2[1] = np.mean(dict[1][:,0]),np.mean(dict[1][:,1])
    m3[0],m3[1] = np.mean(dict[2][:,0]),np.mean(dict[2][:,1])
    print(m1,m2,m3)"""


dict = {}
for i in range(2):
    dict = classify(x,mu)
    plotpoints(dict,i)
    mu = updateMean(dict,mu)
    plotpointsWithM(dict,mu,i)

####################


img = cv2.imread("baboon.jpg")
#print(img.shape)

a = np.reshape(img,(512**2,3))
#print(a.shape)

def dist1(p1, p2):
    return math.sqrt(math.pow((p1[0] - p2[0]),2) + math.pow((p1[1] - p2[1]),2) + math.pow((p1[2] - p2[2]),2))

def classify1(x,mu):
    dict = {}
    for j in range(len(mu)):
        dict[j] = []

    for xy in x:
        """d1 = dist(m1,xy)
        d2 = dist(m2,xy)
        d3 = dist(m3,xy)
        d = [d1,d2,d3]"""
        d = []
        for m in mu:
            d.append(dist1(m,xy))
        index = d.index(min(d))
        dict[index].append(xy)

    for j in range(len(mu)):
        dict[j] = np.asarray(dict[j])
    return dict

def updateMean1(dict,mu):
    #updating the Mu1,2,3 by finding the mean
    for i in range(len(mu)):
        #mu[i][0],mu[i][1],mu[i][2] = np.mean(dict[i][:,0]),np.mean(dict[i][:,1]),np.mean(dict[i][:,2])
        print(len(dict[0]))
        mu[i][0] = np.mean(dict[i][:,0])
        mu[i][1] = np.mean(dict[i][:,1])
        mu[i][2] = np.mean(dict[i][:,2])
    return mu

def applyMeanValues(x,mu):
    for i in range(len(x)):
        d = []
        for m in mu:
            d.append(dist1(m,x[i]))
        index = d.index(min(d))
        x[i] = mu[index]
    return x

ks = [3,5,10,20]

for k in ks:

    mu = [[random.random() for i in range(3)]  for l in range(k)]
    print(mu)

    dict = {}
    for i in range(20):
        dict = classify1(a,mu)
        mu = updateMean1(dict,mu)
    print("updated mu")
    print(mu)
    res = applyMeanValues(a,mu)
    resultImg  = np.reshape(res,(512,512,3))
    cv2.imwrite('task3_baboon_' + str(k) + '.jpg',resultImg)
