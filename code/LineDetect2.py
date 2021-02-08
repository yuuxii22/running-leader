import numpy as np
import cv2 as cv

import math
from sklearn.cluster import AgglomerativeClustering

#Image Dimension
HEIGHT = 480
WIDTH = 640

##Hough Line Parameters
MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 30

##Detect line filter
#Angle from top left origin in degrees
MIN_ANGLE = math.pi * 15 / 180
MAX_ANGLE = math.pi * 165 / 180

#Intercept with bottom (x0,HEIGHT) in pixels
MIN_INTERCEPT = -500
MAX_INTERCEPT = WIDTH+500

##Cluster Threshold
DISTANCE_THRESHOLD = 100

##Dimension Converstion
METER_PER_PIXEL = 0.005

##Camera X Offset
CAMERA_OFFSET = 0

##Single Side Following in pixel
DISTANCE_FROM_LINE = 100

roiVert = np.array([[0,HEIGHT], [0,HEIGHT*2/3], [WIDTH/3, HEIGHT/3], [WIDTH*2/3, HEIGHT/3], [WIDTH,HEIGHT*2/3], [WIDTH,HEIGHT]], dtype=np.int32).reshape(-1,1,2)

def generateLineParam(img):
    global HEIGHT, WIDTH, METER_PER_PIXEL
    global roiVert
    global MIN_ANGLE, MAX_ANGLE
    global MIN_INTERCEPT, MAX_INTERCEPT
    global DISTANCE_THRESHOLD

    #Convert to gray
    imgG = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    #Generate ROI mask
    mask = np.zeros_like(imgG, dtype=np.uint8)
    mask = cv.fillPoly(mask, [roiVert], 255)

    #Blur and Canny
    imgB = cv.GaussianBlur(imgG, (3,3), 0)
    imgC = cv.Canny(imgB, 125, 350)

    #Apply ROI mask
    roi = cv.bitwise_and(imgC, imgC, mask=mask)

    #Detech lines using Prob Hough
    lines = cv.HoughLinesP(roi, 3, math.pi/180, 100, minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)

    if(lines is None):
        return np.array([]), np.array([])

    #print(lines.shape)

    #Filter Lines
    linesList = lines.squeeze(axis=1)
    dy = linesList[:,3] - linesList[:,1]
    dx = linesList[:,2] - linesList[:,0]
    angle = np.arctan2(dy, dx)
    angle[angle < 0] += math.pi

    #Angle Filter
    angleFilt = np.bitwise_and(np.abs(angle) >= MIN_ANGLE, np.abs(angle) <= MAX_ANGLE)
    angle = angle[angleFilt]

    #Intercept Filter
    slope = np.tan(angle)
    intercept = np.divide(HEIGHT - linesList[angleFilt, 1], slope) + linesList[angleFilt, 0]
    interceptFilt = np.bitwise_and(intercept >= MIN_INTERCEPT, intercept <= MAX_INTERCEPT)

    angle = angle[interceptFilt]
    intercept = intercept[interceptFilt]

    #Clustering
    features = np.stack([intercept, angle], axis=-1)

    if(len(features) > 1):

        cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=DISTANCE_THRESHOLD)
        cluster.fit(features)

        clusterCenters = []
        for i in np.unique(cluster.labels_):
            mask = (cluster.labels_ == i)
            clusterCenters.append(np.average(features[mask], axis=0))

        clusterCenters = np.array(clusterCenters)
    
    else:
        clusterCenters = features

    #Separate Cluster into left and right and sort closest to middle first
    clustersInd = np.argsort(clusterCenters[:,0], axis=0)
    clusterSort = clusterCenters[clustersInd]

    leftClusters = clusterSort[clusterSort[:,0] < WIDTH/2][::-1]
    rightClusters = clusterSort[clusterSort[:,0] > WIDTH/2]

    return leftClusters, rightClusters

def calculateSteering(leftClusters, rightClusters, velocity=None, timeToReach=None):
    global HEIGHT, WIDTH, METER_PER_PIXEL
    global DISTANCE_FROM_LINE
    global CAMERA_OFFSET

    #Calculate Target X
    if(len(rightClusters) > 0 and len(leftClusters) > 0):
        targetX = (rightClusters[0,0] + leftClusters[0,0]) / 2
    elif(len(rightClusters) > 0):
        targetX = rightClusters[0,0] - DISTANCE_FROM_LINE
    elif(len(leftClusters) > 0):
        targetX = leftClusters[0,0] + DISTANCE_FROM_LINE
    else:
        #print("No line found")
        return None
        pass

    errX = targetX - (WIDTH/2 + CAMERA_OFFSET)
    errXMeter = METER_PER_PIXEL * errX

    #print(f"Target: {targetX} Error: {errX}")

    #Calculate Angle
    if(velocity is not None and timeToReach is not None):
        d = velocity*timeToReach
        if(d > errXMeter):
            angleRad = math.asin(errXMeter/d)
        else:
            angleRad = 0
    else:
        angleRad = math.asin(errX/max((WIDTH/2),errX))
    
    #print(angleRad)

    return angleRad

def lineParamToCoord(lineParam, lineLength=150):
    global HEIGHT, WIDTH
    #Line Length in pixels
    
    if(len(lineParam) == 0):
        return None

    #Return [x0, y0, x1, y1]
    x0 = lineParam[:,0].copy()
    
    lowX = x0 < 0
    highX = x0 > WIDTH
    
    slope = np.tan(lineParam[:,1])
    
    y0 = HEIGHT*np.ones_like(x0) 
    
    y0[lowX] = HEIGHT - np.divide(x0, slope)[lowX]
    x0[lowX] = 0
    
    y0[highX] = HEIGHT - np.divide(x0-WIDTH,slope)[highX]
    x0[highX] = WIDTH
    
    x1 = x0 - lineLength*np.cos(lineParam[:,1])
    y1 = y0 - lineLength*np.sin(lineParam[:,1])
    
    return np.expand_dims(np.floor(np.stack([x0, y0, x1, y1], axis=-1)).astype(np.int32), 1)

def drawLines(lines, color=(0,255,0), thickness=3, img=None):
    global HEIGHT, WIDTH
    lineImg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    if(img is not None):
        lineImg = img
    
    for l in lines:
        x0,y0,x1,y1 = l[0]
        lineImg = cv.line(lineImg, (x0,y0), (x1,y1), color, thickness)

    return lineImg

def drawLinesCrop(lines, color=(0,255,0), thickness=3, img=None):
    global WIDTH, HEIGHT
    minW = min(lines[:,:,0].min(),lines[:,:,2].min(), 0)
    maxW = max(lines[:,:,0].max(),lines[:,:,2].max(), WIDTH)
    
    minH = min(lines[:,:,1].min(),lines[:,:,3].min(), 0)
    maxH = max(lines[:,:,1].max(),lines[:,:,3].max(), HEIGHT)
    
    #print(maxH, minH, maxW, minW)
    
    lineFound = np.zeros((maxH-minH,maxW-minW,3), dtype=np.uint8)

    for l in lines:
        x0,y0,x1,y1 = l[0]
        lineFound = cv.line(lineFound, (x0-minW,y0-minH), (x1-minW,y1-minH), (0,255,0), 3)
        
        
    lineFound = lineFound[-minH:-minH+HEIGHT,-minW:-minW+WIDTH]
    if(img is not None):
        lineFound = cv.add(lineFound, img)
        
    return lineFound

def drawSteering(angle, lineLength=100, color=(255,255,255), thickness=3, img=None):
    global HEIGHT, WIDTH

    if(angle is None):
        return cv.putText(img, "Angle: None", (10,25), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness=thickness)

    params = lineParamToCoord(np.array([[WIDTH/2, angle+math.pi/2]]), lineLength=lineLength)

    img = drawLines(params, color=color, img=img)
    return cv.putText(img, f"Angle: {180*angle/math.pi:.1f}", (10,25), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness=thickness)

def drawAll(lLines, rLines, angle, img):
    dispImg = img.copy()
    if(lLines is not None):
        dispImg = drawLines(lLines, color=(0,150,0), img=dispImg)
    if(rLines is not None):
        dispImg = drawLines(rLines, color=(0,0,150), img=dispImg)
    dispImg = drawSteering(angleRad, lineLength=150, color=(255,255,255), img=dispImg)
    dispImg = cv.cvtColor(dispImg, cv.COLOR_RGB2BGR)
    return dispImg

if __name__ == "__main__":
    
    import sys

    if(len(sys.argv) == 2):
        img = cv.imread(sys.argv[1])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        lClusters, rClusters = generateLineParam(img)
        angleRad = calculateSteering(lClusters, rClusters)

        lLines = lineParamToCoord(lClusters,lineLength=150)
        rLines = lineParamToCoord(rClusters,lineLength=150)

        dispImg = drawAll(lLines, rLines, angleRad, img)
        cv.imshow("Result", dispImg)

        cv.waitKey(0)
        
    else:
        camera = cv.VideoCapture(0)
        
        print(f"Camera: {camera}")

        while True:
            ret, img = camera.read()
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            lClusters, rClusters = generateLineParam(img)
            angleRad = calculateSteering(lClusters, rClusters)

            lLines = lineParamToCoord(lClusters,lineLength=150)
            rLines = lineParamToCoord(rClusters,lineLength=150)

            dispImg = drawAll(lLines, rLines, angleRad, img)
            cv.imshow("Result", dispImg)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cv.destroyAllWindows()
