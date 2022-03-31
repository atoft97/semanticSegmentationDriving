import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import os
dirname = os.path.dirname(__file__)
import time
#inputStartPath = "segmentetImages/RGB/roadDrive2"
#outputStartPath = "drivableImages/roadDrive2"

#innputFilenames = os.listdir(inputStartPath)
#innputFilenames.sort()

#4:obstacle
#3:Verry rough terrain
#2:rough terrain
#1:Good Terrain
#0:Unkown
class ClassesToDrivable:

    def __init__(self):
        self.color_to_label = {(255, 255, 255): 0, (64, 255, 38): 2, (70, 70, 70): 0, (150, 0, 191): 4, (255, 38, 38): 4, (232, 227, 81): 3, (255, 179, 0): 3, (255, 20, 20):4 , (191, 140, 0):1 , (15, 171, 255):0 , (200, 200, 200): 4, (46, 153, 0): 4, (180, 180, 180): 1}
        self.label_to_color = {0: (0,0,0), 1: (0,170,0), 2: (250,250,0), 3: (250,150,0), 4: (255,0,0)}


    def imageClassesToDrivable(self, image, filename, loggingFolder = ""):
        
        #for filename in tqdm(innputFilenames):
        #filePath = "test.png"
        #image = Image.open(filePath)
        #print(type(image))
        
        image = cv2.cvtColor(image.astype('float32'), cv2.COLOR_BGR2RGB)
        numpyImage = np.asarray(image)
        drivable = np.zeros((numpyImage.shape[0], numpyImage.shape[1]))
        startTime = time.time()
        for key in self.color_to_label.keys():
            drivable[(numpyImage == key).all(2)] = self.color_to_label[key]
        
        diffTime = time.time() - startTime
        #print("bildekovertering inni:", diffTime)

        #print(drivable.shape)

        #cv2.imwrite(indexOutputPath, drivable)

        if (loggingFolder != ""):
            visable = np.zeros((*numpyImage.shape,))
            for key in self.label_to_color.keys():
                visable[drivable == key] = self.label_to_color[key]
            #indexOutputPath = os.path.join(dirname, loggingFolder + "/indexes/" + filename)
            visable = cv2.cvtColor(visable.astype('float32'), cv2.COLOR_BGR2RGB)
            colorOutputPath = os.path.join(dirname, loggingFolder + "/colors/" + filename)
            cv2.imwrite(colorOutputPath, visable)
        else:
            visable = None

        return(drivable, visable)

    
    