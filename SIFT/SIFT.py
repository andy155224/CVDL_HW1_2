import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class SIFT():

    def __init__(self):
        self.image1 = ''
        self.image2 = ''

    def LoadImage1(self, path):
        self.image1 = path
        
    def LoadImage2(self, path):
        self.image2 = path

    def Keypoints(self):
        img = cv2.imread(self.image1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        img = cv2.drawKeypoints(gray, kp, img)

        cv2.imwrite('./Result/Keypoints/result.jpg', img)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def MatchedKeypoints(self):
        img1 = cv2.imread(self.image1)
        img2 = cv2.imread(self.image2)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift1 = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift1.detectAndCompute(img1, None)
        img1 = cv2.drawKeypoints(gray1, kp1, img1)

        sift2 = cv2.xfeatures2d.SIFT_create()
        kp2, des2 = sift2.detectAndCompute(img2, None)
        img2 = cv2.drawKeypoints(gray2, kp2, img2)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        good = np.expand_dims(good,1)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good[:],None, flags=2)

        cv2.imwrite('./Result/MatchedKeypoints/result.jpg', img3)
        cv2.imshow('img', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()