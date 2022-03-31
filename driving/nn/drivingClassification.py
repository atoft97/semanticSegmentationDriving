import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader

import csv
from torch import nn
import torch

class DrivingClassification:

    def __init__(self, indexImagePath, csvPath):
        self.indexImagePath = indexImagePath
        self.csvPath = csvPath


        #self.train_features, self.data, self.train_labels = next(iter(self.dataloader))
        #print(self.train_features)
        #print(self.data)
        #print(self.train_labels)

        self.model = NNDriving().to('cuda')
        print(self.model)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

    def loadData(self):
        self.inputTargetData = self.readCSVdataset(self.csvPath) 
        self.namePosDirection = self.loadPosDirJson()

        self.inputStartPath = self.indexImagePath
        self.innputFilenames = os.listdir(self.inputStartPath)
        self.innputFilenames.sort()

        rgbImageIndex, targets, strights = self.populateDataLists()
        self.dataset = SegmentedDataset(rgbImageIndex, targets, strights)
        self.dataloader = DataLoader(self.dataset, batch_size = 8)


    def readCSVdataset(self, csvPath):#legg på csv input path
        rows = []
        with open(csvPath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                rows.append(row)
        return(rows)
    
    def loadPosDirJson(self): #add path
        with open("data/RGBnamePosDir.json") as inputFile:
            namePosDirection = json.load(inputFile)
        return(namePosDirection)

    def getPos(self, filename):
        return(self.namePosDirection[filename]['latitude'], self.namePosDirection[filename]['longitude'])

    def getDir(self, filename):
        return(self.namePosDirection[filename]['direction']) 

    def populateDataLists(self):
        data = []
        targets = []
        strights = []
        counter = 0

        #print(self.inputTargetData)
        for row in tqdm(self.inputTargetData):
            #print(row)
            filename = row[0]
            straingDirection = float(row[1])
            recomendedDirection = float(row[2])

        #for filename in tqdm(innputFilenames[0:]):
            counter += 1
            filePath = self.inputStartPath + "/" + filename
            image = Image.open(filePath)
            numpyImage = np.asarray(image)
            #print(numpyImage.shape)
            
            data.append(numpyImage) #downscale image first
            #target = 0
            targets.append(recomendedDirection)
            strights.append(straingDirection)

            #break
            #if (counter > 100):
            #    break
        return(data, targets, strights)
    
    def train(self):
        for i in range(100):
            size = len(self.dataloader.dataset)
            self.model.train()
            for batch, (image, data, y) in enumerate(self.dataloader):
                #print(image.shape)
                image = image.float() 
                data = data.float()
                #y = y.long()
                #y = y.long()
                

                image, data, y = image.to('cuda'), data.to('cuda'), y.to('cuda')
                #print(y)

                # Compute prediction error
                pred = self.model(image, data)
                #print(pred.shape)
                #pred = pred[None, :]
                #print(pred.shape)

                #print(y)
                loss = self.loss_fn(pred.to(torch.float32), y.to(torch.float32))
                # Backpropagation
                
                self.optimizer.zero_grad()


                loss.backward()
                self.optimizer.step()

                
                #print(batch)
                #if batch % 100 == 0:
                #    loss, current = loss.item(), batch * len(image)
                    #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print('loss', "turn:", i, {loss.item()})

class SegmentedDataset(Dataset):
    def __init__(self, data, targets, strights, transform=None, imageWith=256):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.strights = strights

        self.angleTargetDict = {}
        numberOfChecks = 7
        fov = 60
        width = int(imageWith/ 4)
        widthPlus = int((imageWith - width)/(numberOfChecks-1))
        degreesPlus = round(widthPlus * (fov/imageWith),1)
        degreesCamera = -fov/2 + (int(width * (fov/imageWith))/2)
        for i in range(numberOfChecks):
            self.angleTargetDict[degreesCamera] = i
            #print(degreesCamera)
            degreesCamera += degreesPlus

    def __getitem__(self, index):
        dataPoint = self.data[index]
        target = self.targets[index]
        stright = self.strights[index]
        x = []
        #split i 4 channels
        for classNumber in range(0, 5):
            class_tensor = np.where(dataPoint == classNumber, 1, 0)
            x.append(class_tensor)
        x_np = np.asarray(x)

        #target = int(round(target,0))
        #y = np.zeros(61)
        #y[target+30] = 1
        y = np.zeros(7)
        y[self.angleTargetDict[target]] = 1
        #y = target
        #print(target)
        #print(y)
        #print(fail)
        
        
        
        stright = stright/360 #normaliser til mellom -1 og 1 pa forhand
        string_np = np.asarray([stright])
    
        return(x_np, string_np, y)
    
    def __len__(self):
        return(len(self.data))




class NNDriving(nn.Module):
    def __init__(self):
        super(NNDriving, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(3, 3, kernel_size=3, stride=3, padding=1),
            nn.Conv2d(3, 3, kernel_size=3, stride=3, padding=1),
            #nn.BatchNorm2d(3),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(3),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.flatten = nn.Flatten()


        self.cnnToLinear = nn.Sequential(
            nn.Linear(841, 400),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
        )
        '''
        self.cnnToLinear = nn.Sequential(
            nn.Linear(65536, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 20)
        )
        '''
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, 7),
            nn.ReLU(),
            nn.Linear(7, 7),
            #nn.ReLU(),
            #nn.Linear(7, 1),
            #nn.Softmax(),
        )

        #self.outputSoftmax = nn.Softmax(dim=1)

        #self.fc1 = nn.Linear(20 + 10, 60)
        #self.fc2 = nn.Linear(60, 5)

    def forward(self, image, data):
        #print("flott")
        x1 = self.cnn_layers(image)
        #print("bra")
        x1 = self.flatten(x1)
        x1 = self.cnnToLinear(x1)
        
        #print(x1)
        #print(failt)
        #x1 = x1[0]
        #liner lag for a minske storrelsen

        x2 = data


        x = torch.cat((x1, x2), dim=1)
        
        #print("2")
        x = self.linear_relu_stack(x)
        #print(x)
        #softmax

        #x = self.outputSoftmax(x)
        
        return x






#loss_fn = nn.CrossEntropyLoss()









