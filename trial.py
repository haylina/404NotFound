
# -*- coding:utf-8 -*-

"""
@author: Jun
@file: .py
@time: 3/5/20184:23 PM
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import cv2
from skimage import morphology,feature,filters
from scipy import ndimage as ndi

def crop_minAreaRect(img,rect):
    # crop the minAreaRectangle image
    (rows, cols) = img.shape
    center = (rows / 2, cols / 2)
    #move of the minAreaRectangle to the center of img
    v_col=32-int(rect[0][0])    #v_col x axis move
    v_row=32-int(rect[0][1])    #v_col y axis move
    M1 = np.float32([[1, 0, v_col], [0, 1, v_row]])
    img = cv2.warpAffine(img, M1, (cols, rows))
    cv2.imshow('img2', img)

    #rotate of the image
    angle=rect[2]
    if rect[1][0]>rect[1][1]:
        angle=(90+angle)
    else:
        angle=angle

    M=cv2.getRotationMatrix2D(center,angle,1.0)
    rot_img=cv2.warpAffine(img,M,(cols,rows),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_TRANSPARENT)
    # cv2.imshow('img3', rot_img)   #show the digit with largest encompassed area

    # rotate bounding box
    box = cv2.boxPoints(rect)
    box1=cv2.transform(np.array([box]), M1)[0]
    pts = np.int0(cv2.transform(np.array([box1]), M))[0]
    # pts[pts < 0] = 0
    # img_3 = cv2.drawContours(rot_img, [pts], 0, (255, 0, 0), 1)
    # img_3 = cv2.resize(img_3, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('img3', img_3)   #show the digit with largest encompassed area

    # crop
    img_crop = rot_img[pts[2][1]:pts[0][1],
                       pts[1][0]:pts[3][0]]
    return img_crop

def PreProcessing():
    i=0
    x=np.zeros((50000,24,15))
    # size=[]
    filename1="D:\\Machine learning projects\\Digit MINIST\\train_x.csv"
    with open(filename1,'r',encoding='utf8') as f:
        for line in f.readlines():
            img=np.array(line.split(','))
            img=img.astype(np.float)
            img=np.int0(img)
            np.savetxt("D:\\Machine learning projects\\Digit MINIST\\try.csv",img,fmt='%d',delimiter=',')
            img=img.reshape(64, 64) # reshape
            img = img.astype(np.uint8)
            # img = filters.rank.median(img, morphology.disk(1))  # 过滤噪声
            ret, thresh = cv2.threshold(img, 230, 255,0)  # Binaryzation: set pixel over 220 to 255; and pixel  below 220 to 0
            thresh = cv2.medianBlur(thresh, 3)  # Image Filtering

            plt.imshow(thresh, cmap=plt.cm.gray)  # to visualize only
            plt.show()
            img2, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # Get the contours in the image
            # cv2.findContours() return: im2, contours, hierarchy

            area0 = 0
            for cont in contours:           #get the contour and minAreaRectangle with the largest area
                rect1 = cv2.minAreaRect(cont)  # cv2.minAreaRect() returns: (center(x, y), (width, height), angle of rotation)
                # if i==43382:
                #     box = cv2.boxPoints(rect1)  # cv2.boxPoints(rect) return: 4 corners of the rectangle
                #     box = np.int0(box)
                #     img_3 = cv2.drawContours(thresh, [box], 0, (255, 0, 0), 1)
                #     img_3 = cv2.resize(img_3, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
                #     cv2.imshow('img3', img_3)   #show the digit with largest encompassed area
                area=rect1[1][0]*rect1[1][1]
                if area>area0:
                     rect=rect1; contour=cont; area0=area


            # for cont in contours:
            #     area = cv2.contourArea(cont)
            #     if area > area0:
            #         contour = cont;
            #         area0 = area
            # rect = cv2.minAreaRect(cont)
            rot_img = crop_minAreaRect(thresh, rect)
            # size.append(rot_img.shape)
            # i+=1
            # print(i)
            rot_img = cv2.resize(rot_img, (15, 24), interpolation=cv2.INTER_CUBIC)
            _,rot_img = cv2.threshold(rot_img, 250, 255, 0)
            cv2.imshow('img3', thresh)
            plt.imshow(rot_img, cmap=plt.cm.gray)  # to visualize only
            plt.show()
            x[i] = rot_img
            print(i)
            i+=1
        f.close()
        x=x.reshape(50000,360)
        # av_row = np.mean([x[0] for x in size])
        # av_col = np.mean([x[1] for x in size])
        # print("av_row",av_row)
        # print("av_col",av_col)
        np.savetxt("D:\\Machine learning projects\\Digit MINIST\\train_x1.csv",x,fmt='%d',delimiter=',')

PreProcessing()


# Machine learning
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.externals import joblib
from time import clock

start=clock()
print("Start: " + str(start))

train_x=np.loadtxt("D:\\Machine learning projects\\Digit MINIST\\train_x1.csv",dtype=int,delimiter=',')
train_y=np.loadtxt("D:\\Machine learning projects\\Digit MINIST\\train_y.csv",dtype=int,delimiter=',')

X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2,random_state=4)
#Linear SVM Classifier
lsvm=LinearSVC()
lsvm.fit(X_train, y_train)
print("Linear SVM Classifier")
print(lsvm.score(X_valid, y_valid))
joblib.dump(lsvm,"D:\\Machine learning projects\\Digit MINIST\\model\\lsvm.pkl" )
stop1=clock()
print("stop_LSVM: " + str(stop1))
print(str(stop1-start) + "seconds")
# scores = cross_val_score(lsvm, train_x, train_y, cv=5, scoring='accuracy')
# print('Linear SVM accuracy:',scores)
# print('Average score:',scores.mean())

#Random Forest classifier
rf=RandomForestClassifier()
rf.fit(X_train, y_train)
print("Random Forest Classifier")
print(rf.score(X_valid, y_valid))
joblib.dump(rf,"D:\\Machine learning projects\\Digit MINIST\\model\\rf.pkl" )
# scores = cross_val_score(lsvm, train_x, train_y, cv=5, scoring='accuracy')
# print('Random Forest accuracy:',scores)
# print('Average score:',scores.mean())
stop2=clock()
print("stop_rf: " + str(stop2))
print(str(stop2-stop1) + "seconds")

#kernel SVC
clf = SVC(kernel='rbf', C=1)
clf.fit(X_train, y_train)
clf.score(X_valid, y_valid)
print('kernel SVC')
print(clf.score(X_valid, y_valid))
joblib.dump(clf,"D:\\Machine learning projects\\Digit MINIST\\model\\clf.pkl" )
stop4=clock()
print("stop_ksvc: " + str(stop4))
print(str(stop4-start) + "seconds")

#kNN
k_scores = []
for k in range(6,10,1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score_=knn.score(X_valid, y_valid)
    # scores = cross_val_score(knn, train_x, train_y, cv=10, scoring='accuracy')
    k_scores.append(score_)
    print('k=',k)
    print(score_)
plt.plot(range(6,10,1), k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
stop3=clock()
print("stop_knn: " + str(stop3))
print(str(stop3-start) + "seconds")
