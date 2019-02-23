import cv2
import numpy as np
import imutils

def task3():
    template = cv2.imread("template.png",0)
    #img = cv2.imread("pos_1.jpg",0)

    imageNames=[]
    for count in range(1,16):
        imageNames.append("pos_"+str(count)+".jpg")
    #print(template)
    #print(len(template))     39
    #print(len(template[0]))  29
    for imgName in imageNames:
        img = cv2.imread(imgName,0)
        GB = cv2.GaussianBlur(img,(3,3),1)
        LI = cv2.Laplacian(GB, cv2.CV_32F)
        #cv2.imwrite("LI.jpg",LI)
        list1 = np.linspace(0.5,1.0,20)
        print(list1)
        """a = (cv2.GaussianBlur(template,(3,3),1))
        cv2.imshow("oct1WithKeyPoints1",template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("oct1WithKeyPoints1",a)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

        #print(template.shape)
        #print(int(template.shape[1] * list1[0]))
        maximum = 0
        x = 0
        y = 0
        tx = 0
        ty = 0
        for k in list1:

            resized = imutils.resize(template, width= int(template.shape[1] * k))
            GB_template = cv2.GaussianBlur(resized,(3,3),1)
            LI_template = cv2.Laplacian(GB_template, cv2.CV_32F)
            result = cv2.matchTemplate(LI, LI_template , cv2.TM_CCOEFF_NORMED)
            a = (cv2.minMaxLoc(result))
            print(a)
            print(resized.shape)
            if (a[1] > maximum):
                maximum = a[1]
                x,y = a[3]
                tx,ty = resized.shape


        #print("max = " , maximum)
        #print("x" ,x)
        #print("y" ,y)
        #print("tx" , tx)
        #print("ty" , ty)
        startX = x
        startY = y
        endX = x + tx
        endY = y + ty

        #print(startX)
        #print(startY)
        #print(endX)
        #print(endY)
        img = cv2.imread(imgName)
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.imwrite("output" + imgName,img)

#######

def bonus():
    template = cv2.imread("template.png",0)
    imageNames=[]
    for count in range(1,11):
        imageNames.append("neg_"+str(count)+".jpg")
    #print(template)
    #print(len(template))     39
    #print(len(template[0]))  29
    for imgName in imageNames:
        img = cv2.imread(imgName,0)
        GB = cv2.GaussianBlur(img,(3,3),1)
        LI = cv2.Laplacian(GB, cv2.CV_32F)
        #cv2.imwrite("LI.jpg",LI)
        list1 = np.linspace(0.5,1.0,20)
        print(list1)
        """a = (cv2.GaussianBlur(template,(3,3),1))
        cv2.imshow("oct1WithKeyPoints1",template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("oct1WithKeyPoints1",a)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

        #print(template.shape)
        #print(int(template.shape[1] * list1[0]))
        maximum = 0
        x = 0
        y = 0
        tx = 0
        ty = 0
        for k in list1:

            resized = imutils.resize(template, width= int(template.shape[1] * k))
            GB_template = cv2.GaussianBlur(resized,(3,3),1)
            LI_template = cv2.Laplacian(GB_template, cv2.CV_32F)
            result = cv2.matchTemplate(LI, LI_template , cv2.TM_CCOEFF_NORMED)
            a = (cv2.minMaxLoc(result))
            print(a)
            print(resized.shape)
            if (a[1] > maximum):
                maximum = a[1]
                x,y = a[3]
                tx,ty = resized.shape


        #print("max = " , maximum)
        #print("x" ,x)
        #print("y" ,y)
        #print("tx" , tx)
        #print("ty" , ty)
        startX = x
        startY = y
        endX = x + tx
        endY = y + ty

        #print(startX)
        #print(startY)
        #print(endX)
        #print(endY)
        if (maximum != 0):
            img = cv2.imread(imgName)
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.imwrite("output" + imgName,img)
        else:
            img = cv2.imread(imgName)
            cv2.imwrite("output" + imgName,img)

task3()
bonus()
