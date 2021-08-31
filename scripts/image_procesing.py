#%%
import os
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%
def imshow(img, cmap=None, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    plt.show()

def imshowpair(img1, img2, figsize=(12,7)):
    plt.figure(figsize=figsize)
    plt.subplot(121),plt.imshow(img1,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img2,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def CannyThreshold(img, low_threshold=75, ratio=3, ksize=3):
    '''
        Canny Edge Detection
        
        input:
            - img: gray image (2 channels)
        output
            -
    '''
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(imgray, (7,7), 0)
    edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, ksize)
    # ret, thresh = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)
    return np.where(edges != 0, 0, 255)

def ContourDetector(img, method="tresh", convexhull=False):
    '''
        Contour Detector
        
        input:
            - img: gray image (2 channels)
            - method: 
                - "tresh" for threshold
                - "edge" for canny

        output: 
            - contour image with white background
    '''
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == "tresh":
        _, temp = cv2.threshold(imgray, 100, 255, 255)
    elif method == "edge":
        img_blur = cv2.GaussianBlur(imgray, (5,5), cv2.BORDER_DEFAULT)
        temp = cv2.Canny(img_blur, 75, 225)
    else:
        raise Exception("wrong method, try tresh or edge")

    if convexhull:
        dst = np.ones(img.shape, np.uint8) * 255
        contours, _ = cv2.findContours(temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            hull = cv2.convexHull(contour)
            cv2.drawContours(dst, [hull], 0, (0,0,0), 1)
        return hull
    else:
        newimg = np.ones(img.shape, np.uint8) * 255
        contours, _ = cv2.findContours(temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(newimg, contours, -1, (0,0,0), 1)
        return newimg

def Laplace(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # dst = np.ones(imgray.shape+(3,), np.uint8) * 255
    img_blur = cv2.GaussianBlur(imgray, (7,7), 0)
    laplacian = cv2.Laplacian(imgray, cv2.CV_16S, (21,21))
    # _, temp = cv2.threshold(laplacian, 0, 1, cv2.THRESH_BINARY)
    laplacian = cv2.convertScaleAbs(laplacian)

    return np.ones(imgray.shape, np.uint8) * 255 - laplacian

def main():
    # load testing dataset
    genre_train = pd.read_csv("../dataset/wikiart_csv/genre_train.csv", delimiter=",", names=["path", "genre"])
    
    # Load first portrair image for testing
    img = cv2.imread(os.path.join("../dataset/wikiart/", genre_train[genre_train.genre == 6].path.values[1]))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    edges = CannyThreshold(img, 75)
    contours = ContourDetector(img, "edge")
    laplacian = Laplace(img)

    plt.figure(figsize=(17,7))
    plt.subplot(141),plt.imshow(imrgb)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(142),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(143),plt.imshow(contours,cmap = 'gray')
    plt.title('Contour Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(144),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian Image'), plt.xticks([]), plt.yticks([])
    plt.show()

#%%
if __name__ == "__main__":
    main()
