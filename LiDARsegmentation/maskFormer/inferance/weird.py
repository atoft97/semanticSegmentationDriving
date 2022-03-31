import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
import torch
from os import listdir

from torchvision.transforms import Compose

from detectron2.modeling import build_model

import json
from detectron2.data.datasets import register_coco_panoptic_separated, register_coco_panoptic
from detectron2.projects.deeplab import add_deeplab_config
from mask_former import add_mask_former_config
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch

import nvidia_smi
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print("Total memory:", info.total)
print("Free memory:", info.free)
print("Used memory:", info.used)

modelName = "semanticRGB"
device = torch.device("cuda")

#register_coco_panoptic_separated(name="ffi_test", sem_seg_root="dataset_test/panoptic_stuff_test", metadata={}, image_root="dataset_test/images", panoptic_root="dataset_test/panoptic_test" , panoptic_json="dataset_test/annotations/panoptic_test.json" ,instances_json="dataset_test/annotations/instances_test.json")
register_coco_panoptic_separated(name="ffi_train", metadata={}, sem_seg_root="../../../data/dataset/train/panoptic_stuff_train", image_root="../../../data/dataset/train/images", panoptic_root="../../../data/dataset/train/panoptic_train" , panoptic_json="../../../data/dataset/train/annotations/panoptic_train.json" ,instances_json="../../../data/dataset/train/annotations/instances_train.json")
register_coco_panoptic_separated(name="ffi_val", metadata={}, sem_seg_root="../../../data/dataset/train/panoptic_stuff_train", image_root="../../../data/dataset/train/images", panoptic_root="../../../data/dataset/train/panoptic_train" , panoptic_json="../../../data/dataset/train/annotations/panoptic_train.json" ,instances_json="../../../data/dataset/train/annotations/instances_train.json")

'''
MetadataCatalog.get("ffi_train_stuffonly").set(stuff_classes=['Person', 'Sky', 'Dirtroad', 'Vehicle', 'Forrest', 'CameraEdge', 'Bush', 'Puddle', 'Large_stone', 'Grass', 'Gravel', 'Building'])
MetadataCatalog.get("ffi_train_stuffonly").set(stuff_colors=[[255, 38, 38], [15, 171, 255], [191, 140, 0], [150, 0, 191], [46, 153, 0], [70, 70, 70], [232, 227, 81], [255, 179, 0], [200, 200, 200], [64, 255, 38], [180, 180, 180], [255, 20, 20]])
#MetadataCatalog.get("ffi_train_stuffonly").set(stuff_dataset_id_to_contiguous_id={1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12})

MetadataCatalog.get("ffi_val_stuffonly").set(stuff_classes=['Person', 'Sky', 'Dirtroad', 'Vehicle', 'Forrest', 'CameraEdge', 'Bush', 'Puddle', 'Large_stone', 'Grass', 'Gravel', 'Building'])
MetadataCatalog.get("ffi_val_stuffonly").set(stuff_colors=[[255, 38, 38], [15, 171, 255], [191, 140, 0], [150, 0, 191], [46, 153, 0], [70, 70, 70], [232, 227, 81], [255, 179, 0], [200, 200, 200], [64, 255, 38], [180, 180, 180], [255, 20, 20]])
#MetadataCatalog.get("ffi_val_stuffonly").set(stuff_dataset_id_to_contiguous_id={1:1, 2:10, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:2, 11:11, 12:12})
'''


MetadataCatalog.get("ffi_train_stuffonly").set(stuff_classes=['other', 'Grass', 'CameraEdge', 'Vehicle', 'Person', 'Bush', 'Puddle', 'Building', 'Dirtroad', 'Sky', 'Large_stone', 'Forrest', 'Gravel'])
MetadataCatalog.get("ffi_train_stuffonly").set(stuff_colors=[[255,255,255], [64,255,38], [70,70,70], [150,0,191], [255,38,38], [232,227,81], [255,179,0], [255,20,20], [191,140,0], [15,171,255], [200,200,200], [46,153,0]])

MetadataCatalog.get("ffi_val_stuffonly").set(stuff_classes=['other', 'Grass', 'CameraEdge', 'Vehicle', 'Person', 'Bush', 'Puddle', 'Building', 'Dirtroad', 'Sky', 'Large_stone', 'Forrest', 'Gravel'])
MetadataCatalog.get("ffi_val_stuffonly").set(stuff_colors=[[255,255,255], [64,255,38], [70,70,70], [150,0,191], [255,38,38], [232,227,81], [255,179,0], [255,20,20], [191,140,0], [15,171,255], [200,200,200], [46,153,0], [180,180,180]])




cfg = get_cfg()
add_deeplab_config(cfg)
add_mask_former_config(cfg)
cfg.merge_from_file("configs/ade20k-150/swin/maskformer_swin_tiny_bs16_160k.yaml")
cfg.MODEL.WEIGHTS = "models/" + modelName +".pth"

modelName = "semanticRGB2"

cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 12
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

cfg.DATASETS.TRAIN = ("ffi_train_stuffonly", )
cfg.DATASETS.TEST = ("ffi_val_stuffonly", )

cfg.freeze()

#modelYo = build_model(cfg)
#print()


metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
print(metadata)
#print(fail)

WINDOW_NAME = "WebCamTest"

video_visualizer = VideoVisualizer(metadata, ColorMode.IMAGE)
predictor = DefaultPredictor(cfg)

print(predictor)
#print(fail)

#typeOfFrame = "Video"
typeOfFrame = "Image"
#typeOfFrame = "VideoOptak"

index = 0
#startPath = "../../../data/combinedImagesTaller/plains_drive"
startPath = "inputImages2"

files = listdir(startPath)
files.sort()
typeOfAnalytics = "both"

saving = True

print(cfg)

if (typeOfFrame == "Video"):
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
elif (typeOfFrame == "VideoOptak"):
    cam = cv2.VideoCapture("/home/potetsos/skule/2021Host/prosjekt/dataFFI/ffiBilder/filmer/rockQuarryIntoWoodsDrive.mp4")
    num_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    codec, file_ext = ("mp4v", ".mp4")
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fname = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/ffiBilder/rockQuarryIntoWoodsDrive_Analyzed" + file_ext
    frames_per_second = cam.get(cv2.CAP_PROP_FPS)
    print("FPS", frames_per_second)
    output_file = cv2.VideoWriter(
                    filename=output_fname,
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    frameSize=(4096, 1536),
                    isColor=True,
            )
    print("\n")
    print(output_file)
    print("\n")

def getFrame(index):
    if (typeOfFrame == "Video" or typeOfFrame == "VideoOptak"):
        success, frame = cam.read()
    if (typeOfFrame == "Image"):
        print(str(index) + "/" +str(len(files)))
        frame = read_image(startPath + "/" +files[index], format="BGR")
        index+=1
    return(frame)

def visualise_predicted_frame(frame, predictions):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_visualizer = Visualizer(frame, metadata)
    sem_seg = predictions["sem_seg"]
    frame_visualizer.draw_sem_seg(sem_seg.argmax(dim=0).to('cpu'), area_threshold=None, alpha=0.5)
    vis_frame = frame_visualizer.output
    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
    return(vis_frame)


'''
[
{"supercategory": "Person",     "color": [255,38,38],   "isthing": 0, "id": 4,  "name": "Person"},
{"supercategory": "Sky",        "color": [15,171,255],  "isthing": 0, "id": 8,  "name": "Sky"},
{"supercategory": "Dirtroad",   "color": [191,140,0],   "isthing": 0, "id": 7,  "name": "Dirtroad"},
{"supercategory": "Vehicle",    "color": [150,0,191],   "isthing": 0, "id": 3,  "name": "Vehicle"},
{"supercategory": "Forrest",    "color": [46,153,0],    "isthingmodelFFI": 0, "id": 1,  "name": "Forrest"},
{"supercategory": "CameraEdge", "color": [70,70,70],    "isthing": 0, "id": 2,  "name": "CameraEdge"},
{"supercategory": "Bush",       "color": [232,227,81],  "isthing": 0, "id": 5,  "name": "Bush"},
{"supercategory": "Puddle",     "color": [255,179,0],   "isthing": 0, "id": 6,  "name": "Puddle"},
{"supercategory": "Large_stone","color": [200,200,200], "isthing": 0, "id": 9,  "name": "Large_stone"},
{"supercategory": "Grass",      "color": [64,255,38],   "isthing": 0, "id": 10, "name": "Grass"},
{"supercategory": "Gravel",     "color": [180,180,180], "isthing": 0, "id": 11, "name": "Gravel"},
{"supercategory": "Building",   "color": [255,20,20],   "isthing": 0, "id": 12, "name": "Building"}
]
'''

'''
MetadataCatalog.get("ffi_val_stuffonly").set(stuff_classes=['Building',  'Grass','CameraEdge', 'Vehicle','Person', 'Forrest',  'Bush', 'Puddle', 'Dirtroad','Sky','Large_stone'  'Gravel', 'Forrest'])
MetadataCatalog.get("ffi_val_stuffonly").set(stuff_colors=[[255, 20, 20], [64, 255, 38], [70, 70, 70], [150, 0, 191], [255, 38, 38],[ 46, 153, 0], [232, 227, 81], [255,179,0],  [191, 140, 0],   [15, 171, 255],     [200, 200, 200],  [46,153,0],])
#MetadataCatalog.get("ffi_val_stuffonly").set(stuff_dataset_id_to_contiguous_id={1:1, 2:10, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:2, 11:11, 12:12})
'''
lable_to_color = {}
colorCounter = 0
for color in metadata.get('stuff_colors'):
    lable_to_color[colorCounter] = color
    colorCounter += 1
    #print(color)
#print(fail)

#lable_to_color = {0: [255,20,20], 1:[64, 255, 38], 2: [70, 70, 70], 3: [150, 0, 191], 4: [255, 38, 38], 5: [ 46, 153, 0], 6: [232, 227, 81], 7: [255,179,0], 8: [191, 140, 0], 9: [15, 171, 255],10:[200, 200, 200], 11: [46,153,0]}

#lable_to_color = {9:[0, 0, 255]}


def visEachClass(frame, predictions, fileName):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_visualizer = Visualizer(frame, metadata)
    sem_seg = predictions["sem_seg"]
    #print(sem_seg)
    #print(sem_seg.shape)
    #print(type(sem_seg[0]))
    #pred = (sem_seg[0] * 128).to('cpu').numpy()
    #print(pred)
    #print(len(sem_seg))

    #print(sem_seg[2])
    #print(sem_seg[5])
    #print(sem_seg[11])

    classImage = sem_seg.argmax(dim=0).to('cpu')
    #print(classImage)
    #print(classImage.shape)

    os.makedirs("outputImages/LiDAR/" + modelName + "/" + fileName[:-4], exist_ok=True)

    for classNumber in range(len(sem_seg)):
        pred = (sem_seg[classNumber] * 255).to('cpu').numpy()
        cv2.imwrite("outputImages/LiDAR/" + modelName + "/" + fileName[:-4] + "/" + str(classNumber) + ".png", pred)

    rgb_img = np.zeros((*classImage.shape, 3))
    for key in lable_to_color.keys():
        rgb_img[classImage == key] = lable_to_color[key]
    #print(rgb_img.astype('float32'))
    rgb_img = cv2.cvtColor(rgb_img.astype('float32'), cv2.COLOR_BGR2RGB)
    #print(rgb_img)'
    cv2.imwrite("outputImages/LiDAR/" + modelName + "/" + "outputDatasett" + "/SegmentationClass/" + fileName[:-3] + "png", rgb_img)

'''
# label:color_rgb:parts:actions
background:0,0,0::
Building:32,64,0::
Bush:160,64,128::
CameraEdge:227,55,212::
Dirtroad:255,106,77::
Forrest:144,96,0::
Grass:128,227,21::
Gravel:192,128,32::
Large_stone:7,130,179::
Person:255,0,204::
Puddle:92,244,86::
Sky:50,183,250::
Vehicle:16,160,192::
'''

os.makedirs("outputImages/LiDAR/" + modelName, exist_ok=True)
os.makedirs("outputImages/LiDAR/" + modelName + "/" + "outputDatasett" + "/SegmentationClass", exist_ok=True)

def writeLabelMap():
    os.makedirs("outputImages/LiDAR/" + modelName + "/" + "outputDatasett", exist_ok=True)
    with open("outputImages/LiDAR/" + modelName + "/" + "outputDatasett" + "/labelmap.txt", 'w') as f:
        for i in range(len(metadata.get("stuff_classes"))):
            #print("flott")
            #print(metadata.get("stuff_classes")[i])
            f.write(str(metadata.get("stuff_classes")[i]))
            f.write(":")
            #print(str(metadata.get("stuff_colors")[i]))
            f.write(str(metadata.get("stuff_colors")[i][0]))
            f.write(",")
            f.write(str(metadata.get("stuff_colors")[i][1]))
            f.write(",")
            f.write(str(metadata.get("stuff_colors")[i][2]))
            f.write(":")
            f.write(":")
            f.write('\n')
    
writeLabelMap()

def detectronToCoco(predictions, iamgeID, startID):
    #panoptic_seg, segments_info = predictedPanoptic["panoptic_seg"]
    sem_seg = predictions["sem_seg"]
    classImage = sem_seg.argmax(dim=0).to('cpu')
    tensors = []
    #for segment_number in range(1, (len(segments_info) +1)): #range på alle klassa
    #print(classImage)
    for classNumber in range(1, len(sem_seg) +1):
        #print(pan_class)
        class_tensor = torch.where(classImage == classNumber, 1, 0)
        tensors.append(class_tensor)
        #print(classNumber)
        #print(class_tensor)
        #print("\n")

    #print(tensors)
    #print(fail)
    stacked = torch.stack(tensors)
    stackedNumpy = stacked.cpu().detach().numpy()

    number=10

    annotations = []



    for segment_number in range((len(sem_seg))):
        segmentDict = {'id': segment_number+startID}

        #thing_id = dict((v,k) for k,v in metadata.thing_dataset_id_to_contiguous_id.items())
        #stuff_id = dict((v,k) for k,v in metadata.stuff_dataset_id_to_contiguous_id.items())

        #cat_id = segments_info[segment_number]['category_id']
        #cat_id = segment_number

        #if (segments_info[segment_number]['isthing']):
        #    segmentDict['category_id'] = thing_id[cat_id]
        #else:
        #    segmentDict['category_id'] = stuff_id[cat_id]
        segmentDict['category_id'] = segment_number +1
        
        segmentDict['image_id'] = iamgeID
        segmentDict['area'] = 0
        segmentDict['bbox'] = [0,0,0,0]
        segmentDict['iscrowd'] = 0

        border = cv2.copyMakeBorder((stackedNumpy[segment_number]).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0 )
        contours, hierarchy = cv2.findContours(border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation_list = []


        for countor in contours:
            #epsilon = 0.0003 * cv2.arcLength(countor, True)
            #approximations = cv2.approxPolyDP(countor, epsilon, True)

            x_y_list = []
            #for coordinates in approximatifileNameons:
            for coordinates in countor:
                x_y_list.append(int(coordinates[0][0]))
                x_y_list.append(int(coordinates[0][1]))
            if (len(x_y_list)>4):
                segmentation_list.append(x_y_list)

        segmentDict['segmentation'] = segmentation_list
        annotations.append(segmentDict)
    return(annotations)

def metadatToCoco(metadata):
    categories = []
    for i in range(1, len(metadata.stuff_classes)):
        category = {'id': i, 'name': metadata.stuff_classes[i], "supercategory": ""}
        categories.append(category)
    return(categories)

coco = {"licenses": [
    {
      "name": "",
      "id": 0,
      "url": ""
    }],  
    "info": {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": ""
  },
  }
categories = metadatToCoco(metadata)
coco['categories'] = categories

print(coco)

images = []
segmentID = 1
counter=0
totalTime = 0
annotations = []
for fileName in tqdm(files):
    os.makedirs("outputImages/LiDAR/" + modelName + "/" + "outputDatasett/ImageSets/Segmentation", exist_ok=True)
    with open("outputImages/LiDAR/" + modelName + "/" + "outputDatasett/ImageSets/Segmentation" + "/default.txt", 'a') as f:
        f.write(fileName)
        f.write('\n')

    counter += 1
    #print(str(counter) + "/" +str(len(files)))
    frame = read_image(startPath + "/" +fileName, format="BGR")
    startTime = time.time()
    predictedPanoptic = predictor(frame)
    #print(predictedPanoptic)
    diffTime = time.time() - startTime
    totalTime += diffTime
    #print("avgTime", totalTime/counter)
    vis_panoptic = visualise_predicted_frame(frame, predictedPanoptic)
    visEachClass(frame, predictedPanoptic, fileName)
    combinedFrame = np.vstack((vis_panoptic, frame))
    #print(combinedFrame)
    #print(combinedFrame.shape)

    if (saving == True):
        cv2.imwrite("outputImages/LiDAR/" + modelName + "/" + fileName, combinedFrame)

    
    height, width = frame.shape[:2]
    imageCoco = {"id": counter, "width": width, "height": height, "file_name": fileName[:-3] + "png", "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0}
    images.append(imageCoco)

    annotationsImage = detectronToCoco(predictedPanoptic, counter, segmentID)
    #print(annotationsImage)
    annotations.extend(annotationsImage)
    segmentID += len(annotationsImage)

coco['images'] = images
coco['annotations'] = annotations
with open("outputImages/LiDAR/" + modelName + "/" + "outputDatasett/anotationsFFI.json", 'w') as fp:
    json.dump(coco, fp)