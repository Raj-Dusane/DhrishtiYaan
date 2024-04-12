import cv2
import numpy as np

def warpImg (img,points,w,h,inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp

def drawPoints(img,points):
    for x in range( 4):
        cv2.circle(img,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv2.FILLED)
    return img

def thresholding(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([80, 0, 0])
    upperWhite = np.array([255, 160, 255])
    maskedWhite= cv2.inRange(hsv,lowerWhite,upperWhite)
    return maskedWhite

def nothing(a):
    pass

def initializeTrackbars(intialTracbarVals,wT=480, hT=240):
    # syntax for createTrackbar
    # Parameters:
    #     trackbarName (str): The name to be displayed next to the trackbar.
    #     windowName (str): The name of the OpenCV window where the trackbar will be placed.
    #     minValue (int): The minimum value the trackbar can be set to. Here in this function width and height is by default set to as 480 and 240 respectively.
    #     maxValue (int): The maximum value the trackbar can be set to.
    #     callbackFunc (function): An optional callback function that gets called whenever the trackbar position changes. This function receives the current trackbar position as an argument.
    # cv2.createTrackbar(trackbarName, windowName, minValue, maxValue, callbackFunc)

    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)

def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points

# Creating histogram
def getHistogram(img, minPer=0.1, display=False, region=1):
    # axis = 0 is for finding the sum of each column i.e. in Y-direction and the sum of each column will be respresented by its index.
    # np.sum() function returns the numpy array consisting sum of each columns respectively.
    
    if region==1:  
        # considering full image
        histValues = np.sum(img, axis=0) 
    else :  
        # considering some smaller region of image
        histValues = np.sum(img[img.shape[0]//region:,:], axis=0)
    
    maxValue = np.max(histValues)
    minVal = minPer*maxValue

    indexArray = np.where(histValues >= minVal) # This will return an array which will have the summation greater than the threshold value
    basePoints = int(np.average(indexArray)) # This will average the sum of an array having values greater than thershold value. By default it returns float value.
    # print(basePoints)

    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)  # creating dummy image
        for x, intensity in enumerate(histValues):
            # print(intensity)
            # intensity refers to sum of particular column value

            if intensity > minVal: color=(255, 0, 255)
            else: color=(0, 0, 255)

            cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0]-intensity//255), color,1)
            cv2.circle(imgHist,(basePoints,img.shape[0]),20,(0,255,255),cv2.FILLED)
        return basePoints, imgHist
    
    return basePoints

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver