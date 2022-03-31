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
import os
dirname = os.path.dirname(__file__)

class Segmented2Direction:

    def __init__(self, typeOfImage, csvPath, loggingFolder, goalLatitude =  61.1775, goalLongitude =  11.416):
        self.loggingFolder = loggingFolder
        self.goalLatitude =  goalLatitude
        self.goalLongitude = goalLongitude
        self.typeOfImage = typeOfImage
        
        self.mapImage = Image.open('driving/mapsMap.png', 'r')
        self.BBox = ((11.4040, 11.4262, 61.1740, 61.1871))

        #if (typeOfImage == "lidar"):
        #    with open("data/LiDARnamePosDir.json") as inputFile:
        #        self.namePosDirection = json.load(inputFile)

        #elif (typeOfImage == "rgb"):
        #    with open("data/RGBnamePosDir.json") as inputFile:
        #        self.namePosDirection = json.load(inputFile)
        
        self.counter = 0

        #self.csvPath = datasetName + "-" + str(goalLatitude) + "-" + str(goalLongitude) + ".csv"
        self.csvPath = csvPath


        #inputStartPath = "drivableImages/roadDrive2/indexes"
        #outputStartPath = "outputDirections/roadDrive2"

        #colorStartPath = "segmentetImages/RGB/roadDrive2"

        #innputFilenames = os.listdir(inputStartPath)
        #innputFilenames.sort()

#goalLatitude =  61.17673709034513
#goalLongitude =  11.411760910980844



#goalLatitude =  70
#goalLongitude = 20

#goalLatitude =  38.627089
#goalLongitude =  -90.200203


#st = Goal

    
#if (os.path.exists(csvPath)):
#    os.remove(csvPath)

    def write_to_csv(self, new_data, fileName):
        with open(fileName, 'a') as f_object:
            writer_object = DictWriter(f_object, fieldnames=['imageName', 'strightDirection', 'recomendedDirection'])
            writer_object.writerow(new_data)
            f_object.close()




    def getPos(self, filename):
        return(self.namePosDirection[filename]['latitude'], self.namePosDirection[filename]['longitude'])

    def getDir(self, filename):
        return(self.namePosDirection[filename]['direction']) 


    def checkAreas(self, image, directionChange, rangeImage, useDepth):
        imageSize = image.shape
        print("størrelse", imageSize)
        #print(directionChange)
        
        if (self.typeOfImage == "lidar"):
            numberOfChecks = 359
            fov = 360
            width = int(imageSize[1]/ 10)
        elif (self.typeOfImage == "rgb"):
            #numberOfChecks = 31
            numberOfChecks = 7
            fov = 60
            width = int(imageSize[1]/ 4)
        #antar at kjøretøyet er flatt, tar ikkje hensyn til at pixeler høyt kan være nært i en snart oppoverbakke
        if (useDepth):
            hight = 0, imageSize[0]
        else:
            hight = int(imageSize[0] -(imageSize[0] / 2.8)), int(imageSize[0])

        
        
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

            score = self.getAreaScore(hightStart, hightStop, widthStart, widthStop, image, rangeImage, useDepth)

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

    def getAreaScore(self, hightStart, hightStop, widthStart, widthStop, image, rangeImage, useDepth):
        #print(image[image == 3].shape)
        #print("\n")
        #print("range størrelse", rangeImage.shape)
        #print("ny sørrelse", image.shape)

        

        
        if (useDepth):
            partImage = image
            partImage[rangeImage<100] = 0#alt innafor 5 meter, alternativt, skaler frå range verdien (for kvar pixel gang range verdi med klasseverdi)
    
        partImage = image[hightStart:hightStop, widthStart:widthStop]

        #print("part størrelse", partImage.shape)
        #cv2.imwrite("tmp" + "/"+ str(widthStart) + ".png", partImage*50)
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
        score += countDict[4] * (-10000) #obsticle
        score += countDict[3] * (-10) #verry bad terrain 
        score += countDict[2] * (-1) #rough terrain
        score += countDict[1] * (1) #good terrain
        score += countDict[0] * (0) #unknown

        #print("score", score)
        return(score/1000)


    
    #for filename in tqdm(innputFilenames[2150:]):
    def getDirectionOfImage(self,indexImage, colorImage, fileName, rangeImage, rgb_img, drivableColor, useDepth, lat, long, heading): 
        self.counter += 1
        numpyImage = np.asarray(indexImage) # kanskje unødvendig vist den allerde e numpy
        
        latitude = lat
        longitude = long
        direction = heading

        #latitude, longitude = self.getPos(fileName)
        #direction = self.getDir(fileName)
        latitude = float(latitude)
        longitude = float(longitude)
        direction = float(direction)

        radDirection = math.atan2(  ( math.radians(self.goalLongitude)       -     math.radians(longitude)     )   ,   (  math.radians(self.goalLatitude)   -  math.radians(latitude) ))
        strightDirection = math.degrees(radDirection)

        rad = math.radians(strightDirection)
        x_coord = math.sin(rad)*0.002  + longitude
        y_coord = math.cos(rad)*0.002 + latitude

        rad_actual = math.radians(direction)
        x_coord_actual = math.sin(rad_actual)*0.002  + longitude
        y_coord_actual = math.cos(rad_actual)*0.002 + latitude
        '''
        fig, ax = plt.subplot_mosaic([['1', '2'],
                                        ['1', '3'],
                                        ['1', '4'],
                                        ['1', '5'],
                                        ['1', '6']],
                              figsize=(15,10), constrained_layout=True)
        '''



        direction = direction % 360
        strightDirection = strightDirection % 360

        directionChange = (strightDirection - direction)
        
        bestDegreeLocal, bestBounderies = self.checkAreas(numpyImage, directionChange, rangeImage, useDepth)

        bestDegree = (direction  + bestDegreeLocal) % 360

        rad_best = math.radians(bestDegree)
        x_coord_best = math.sin(rad_best)*0.002  + longitude
        y_coord_best = math.cos(rad_best)*0.002 + latitude

        if (self.loggingFolder != ""):
            
            fig, ax = plt.subplots(nrows=7, figsize = (15,10))
            ax[0].scatter(self.goalLongitude ,self.goalLatitude, zorder=1, c='b', s=10)
            ax[0].scatter(longitude, latitude, zorder=1, c='b', s=10)

            ax[0].set_xlim(11.4040,11.4262)
            ax[0].set_ylim(61.1740,61.1871)

            ax[0].imshow(self.mapImage, zorder=0, extent = self.BBox, aspect= 'equal')
            ax[0].annotate("", xy=(longitude, latitude), xytext=(x_coord, y_coord), arrowprops=dict(arrowstyle="<-", color='blue'))

            ax[0].annotate("", xy=(longitude, latitude), xytext=(x_coord_actual, y_coord_actual), arrowprops=dict(arrowstyle="<-", color='red'))

            ax[0].annotate("", xy=(longitude, latitude), xytext=(x_coord_best, y_coord_best), arrowprops=dict(arrowstyle="<-", color='green'))
            

            #colorFilePath = colorStartPath + "/" + fileName
            #colorImage = Image.open(colorFilePath)
            colorImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB)
            colorNumpyImage = np.asarray(colorImage)
            ax[1].imshow(colorNumpyImage, zorder=0, aspect= 'equal')
            rect = patches.Rectangle((bestBounderies[2], bestBounderies[0]), bestBounderies[3] -bestBounderies[2], bestBounderies[1] - bestBounderies[0], linewidth=1, edgecolor='r', facecolor='none')
            ax[1].add_patch(rect)
            
            if (useDepth):
                rangeNumpyImage = np.asarray(rangeImage)
                rect = patches.Rectangle((bestBounderies[2], bestBounderies[0]), bestBounderies[3] -bestBounderies[2], bestBounderies[1] - bestBounderies[0], linewidth=1, edgecolor='r', facecolor='none')
                ax[2].imshow(rangeNumpyImage, zorder=0, aspect= 'equal')
                ax[2].add_patch(rect)

            
            rect = patches.Rectangle((bestBounderies[2], bestBounderies[0]), bestBounderies[3] -bestBounderies[2], bestBounderies[1] - bestBounderies[0], linewidth=1, edgecolor='r', facecolor='none')
            rgb_img = rgb_img/256
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_imgNumpyImage = np.asarray(rgb_img)
            ax[3].imshow(rgb_img, zorder=0, aspect= 'equal')
            ax[3].add_patch(rect)

            rect = patches.Rectangle((bestBounderies[2], bestBounderies[0]), bestBounderies[3] -bestBounderies[2], bestBounderies[1] - bestBounderies[0], linewidth=1, edgecolor='r', facecolor='none')
            partImageVisual = rgb_img
            partImageVisual[rangeImage<100] = 0
            ax[4].imshow(partImageVisual, zorder=0, aspect= 'equal')
            ax[4].add_patch(rect)

            rect = patches.Rectangle((bestBounderies[2], bestBounderies[0]), bestBounderies[3] -bestBounderies[2], bestBounderies[1] - bestBounderies[0], linewidth=1, edgecolor='r', facecolor='none')
            drivableColor = cv2.cvtColor(drivableColor, cv2.COLOR_BGR2RGB)
            rgb_imgNumpyImage = np.asarray(drivableColor)
            ax[5].imshow(rgb_imgNumpyImage, zorder=0, aspect= 'equal')
            ax[5].add_patch(rect)

            rect = patches.Rectangle((bestBounderies[2], bestBounderies[0]), bestBounderies[3] -bestBounderies[2], bestBounderies[1] - bestBounderies[0], linewidth=1, edgecolor='r', facecolor='none')
            partImageDrivable = drivableColor
            partImageDrivable[rangeImage<100] = 0
            ax[6].imshow(partImageDrivable, zorder=0, aspect= 'equal')
            ax[6].add_patch(rect)

            indexOutputPath = os.path.join(dirname, f'mapPlot10/{fileName}')
            plt.savefig(indexOutputPath)
            plt.close()
        
            strightReconended = {"imageName": fileName, "strightDirection": directionChange, "recomendedDirection": bestDegreeLocal}
            self.write_to_csv(strightReconended, self.csvPath)


        #print(bestBounderies)
        print("faktisk:", direction)
        print("anbefalt:", strightDirection)
        print("change:", directionChange)
        print("bestDegree local:", bestDegreeLocal)
        print("bestDegree:", bestDegree)
        print("\n")



        #del image
        #del colorImage
        #gc.collect()
        
        return(bestDegreeLocal)

        #if (counter == 1):
        #    break

    