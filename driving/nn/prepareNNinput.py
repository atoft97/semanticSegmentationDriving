from itertools import count
import os
from tracemalloc import start
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gc
from csv import DictWriter

inputStartPath = "../drivableImages/roadDrive2/indexes"
outputStartPath = "../outputDirections/roadDrive2"

colorStartPath = "../segmentetImages/RGB/roadDrive2"

innputFilenames = os.listdir(inputStartPath)
innputFilenames.sort()

#goalLatitude =  61.17673709034513
#goalLongitude =  11.411760910980844

goalLatitude =  61.1775
goalLongitude =  11.416

#goalLatitude =  70
#goalLongitude = 20

#goalLatitude =  38.627089
#goalLongitude =  -90.200203


#st = Goal

mapImage = Image.open('../mapsMap.png', 'r')
BBox = ((11.4040, 11.4262, 61.1740, 61.1871))

csvPath = "bagFile-" + str(goalLatitude) + "-" + str(goalLongitude) + ".csv"
#if (os.path.exists(csvPath)):
#    os.remove(csvPath)

def write_to_csv(new_data, fileName):
    with open(fileName, 'a') as f_object:
        writer_object = DictWriter(f_object, fieldnames=['imageName', 'strightDirection', 'recomendedDirection'])
        writer_object.writerow(new_data)
        f_object.close()

with open("../../data/RGBnamePosDir.json") as inputFile:
    namePosDirection = json.load(inputFile)

def getPos(filename):
	return(namePosDirection[filename]['latitude'], namePosDirection[filename]['longitude'])

def getDir(filename):
	return(namePosDirection[filename]['direction']) 


def checkAreas(image, directionChange):
    imageSize = image.shape
    #print(imageSize)
    #print(directionChange)
    numberOfChecks = 31
    fov = 60
    #antar at kjøretøyet er flatt, tar ikkje hensyn til at pixeler høyt kan være nært i en snart oppoverbakke
    hight = int(imageSize[0] -(imageSize[0] / 2.8)), int(imageSize[0])
    width = int(imageSize[1]/ 4)
    widthPlus = int((imageSize[1] - width)/(numberOfChecks-1))

    #print("width", width)
   # print(widthPlus, "widthPlus")
    degreesPlus = round(widthPlus * (fov/imageSize[1]),1)
    #print(degreesPlus*(numberOfChecks-1))
    
   # print(degreesPlus, "plussern")
    #print(feil)
    degreesCamera = -fov/2 + (int(width * (fov/imageSize[1]))/2)
    #print(degreesCamera)
    #print(degreesCamera+degreesPlus*(numberOfChecks-1))
    #print(feil)
    widthStart = 0

    bestScaled = float('-inf')
    bestDegree = 0 
    bestBounderies = ()

    for check in range(numberOfChecks):
        
        
        hightStart = hight[0]
        hightStop = hight[1]
        widthStop = widthStart + width

        score = getAreaScore(hightStart, hightStop, widthStart, widthStop, image)

        offsetDegrees = abs(directionChange-degreesCamera)
        

        scaledScore = score - offsetDegrees*10
        


        #print("deg", degreesCamera)
        #print("score", score)
        #print("offcet", offsetDegrees)
        #print("scaled", scaledScore)
        #print("\n")

        #print("bestScaled", bestScaled)
        if (scaledScore  > bestScaled):
        #    print("yo")
            bestScaled = scaledScore
            bestDegree = degreesCamera
            bestBounderies = (hightStart, hightStop, widthStart, widthStop)

        #split kvart bilde i 4 og gje score etter den 
      #  print(widthStart)
        widthStart += widthPlus
       # print(degreesCamera)
        degreesCamera += degreesPlus
    
    #print("best score", bestScaled)
    #print("best degree", bestDegree)
    return(bestDegree, bestBounderies)

def getAreaScore(hightStart, hightStop, widthStart, widthStop, image):
    #print(image[image == 3].shape)
    #print(hightStart, hightStop, widthStart, widthStop)
    partImage = image[hightStart:hightStop, widthStart:widthStop]
    #print(partImage.shape)
    cv2.imwrite("tmp" + "/"+ str(widthStart) + ".png", partImage*50)
    #print(partImage.shape)
    #unique, counts = np.unique(partImage, return_counts=True)
    #countDict = dict(zip(unique, counts))
    countDict = {}

    countDict[0] = len(partImage[partImage == 0])
    countDict[1] = len(partImage[partImage == 1])
    countDict[2] = len(partImage[partImage == 2])
    countDict[3] = len(partImage[partImage == 3])
    countDict[4] = len(partImage[partImage == 4])

    score = 0
    score += countDict[4] * (-10000)
    score += countDict[3] * (-10)
    score += countDict[2] * (-1)
    score += countDict[1] * (0)

    #print("score", score)
    return(score/1000)


counter = 0
for filename in tqdm(innputFilenames[2527:2820]):
    counter += 1
    filePath = inputStartPath + "/" + filename
    image = Image.open(filePath)
    numpyImage = np.asarray(image)
    #print(numpyImage.shape)

    latitude, longitude = getPos(filename)
    direction = getDir(filename)
    latitude = float(latitude)
    longitude = float(longitude)
    direction = float(direction)
    #latitude = 60
    #longitude = 20

    #latitude = 39.099912
    #longitude = -94.581213

    #print(latitude, longitude, direction)

    #X = math.cos(math.radians(goalLatitude)) * math.sin(math.radians(goalLongitude) - math.radians(longitude))
    #Y = math.cos(math.radians(latitude)) * math.sin(math.radians(goalLatitude)) - math.sin(math.radians(latitude)) * math.cos(math.radians(goalLatitude)) * math.cos(math.radians(goalLongitude) - math.radians(longitude))

    #print("X:", X)
    #print("Y:", Y)

    #radDirection = math.atan2(math.radians(X), math.radians(Y))
    #print("Rad:", radDirection)

    #radDirection = math.tan(  (    math.radians(longitude)     -    math.radians(goalLongitude)     )   /   (  math.radians(latitude)  -  math.radians(goalLatitude) ))

    #radDirection = math.tan(  ( math.radians(goalLongitude)       -     math.radians(longitude)     )   /   (  math.radians(goalLatitude)   -  math.radians(latitude) ))

    radDirection = math.atan2(  ( math.radians(goalLongitude)       -     math.radians(longitude)     )   ,   (  math.radians(goalLatitude)   -  math.radians(latitude) ))

    strightDirection = math.degrees(radDirection)
    #print(strightDirection)

    #print(strightDirection)

    rad = math.radians(strightDirection)
    x_coord = math.sin(rad)*0.002  + longitude
    y_coord = math.cos(rad)*0.002 + latitude

    rad_actual = math.radians(direction)
    x_coord_actual = math.sin(rad_actual)*0.002  + longitude
    y_coord_actual = math.cos(rad_actual)*0.002 + latitude

    fig, ax = plt.subplots(nrows=2, figsize = (15,10))
    ax[0].scatter(goalLongitude ,goalLatitude, zorder=1, c='b', s=10)
    ax[0].scatter(longitude, latitude, zorder=1, c='b', s=10)

    ax[0].set_xlim(11.4040,11.4262)
    ax[0].set_ylim(61.1740,61.1871)

    ax[0].imshow(mapImage, zorder=0, extent = BBox, aspect= 'equal')
    ax[0].annotate("", xy=(longitude, latitude), xytext=(x_coord, y_coord), arrowprops=dict(arrowstyle="<-", color='blue'))

    ax[0].annotate("", xy=(longitude, latitude), xytext=(x_coord_actual, y_coord_actual), arrowprops=dict(arrowstyle="<-", color='red'))

    direction = direction % 360
    strightDirection = strightDirection % 360



    directionChange = (strightDirection - direction)
 


    
    bestDegreeLocal, bestBounderies = checkAreas(numpyImage, directionChange)

    bestDegree = (direction  + bestDegreeLocal) % 360




    rad_best = math.radians(bestDegree)
    x_coord_best = math.sin(rad_best)*0.002  + longitude
    y_coord_best = math.cos(rad_best)*0.002 + latitude

    ax[0].annotate("", xy=(longitude, latitude), xytext=(x_coord_best, y_coord_best), arrowprops=dict(arrowstyle="<-", color='green'))
    

    colorFilePath = colorStartPath + "/" + filename
    colorImage = Image.open(colorFilePath)
    colorNumpyImage = np.asarray(colorImage)
    print(bestBounderies)

    ax[1].imshow(colorNumpyImage, zorder=0, aspect= 'equal')

    #hightStart, hightStop, widthStart, widthStop
    rect = patches.Rectangle((bestBounderies[2], bestBounderies[0]), bestBounderies[3] -bestBounderies[2], bestBounderies[1] - bestBounderies[0], linewidth=1, edgecolor='r', facecolor='none')
    ax[1].add_patch(rect)


    plt.savefig(f'../mapPlot3/{filename}')
    plt.close()


    print("faktisk:", direction)
    print("anbefalt:", strightDirection)
    print("change:", directionChange)
    print("bestDegree local:", bestDegreeLocal)
    print("bestDegree:", bestDegree)
    print("\n")

    strightReconended = {"imageName": filename, "strightDirection": directionChange, "recomendedDirection": bestDegreeLocal}
    write_to_csv(strightReconended, csvPath)

    del image
    del colorImage
    gc.collect()
    


    #if (counter == 1):
    #    break

    