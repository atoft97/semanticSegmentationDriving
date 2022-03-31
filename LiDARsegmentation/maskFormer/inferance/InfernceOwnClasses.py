import argparse
import glob
import multiprocessing as mp
from traceback import print_tb
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

modelName = "semanticRGB"

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print("Total memory:", info.total)
print("Free memory:", info.free)
print("Used memory:", info.used)

device = torch.device("cuda")

#register_coco_panoptic_separated(name="ffi_test", sem_seg_root="dataset_test/panoptic_stuff_test", metadata={}, image_root="dataset_test/images", panoptic_root="dataset_test/panoptic_test" , panoptic_json="dataset_test/annotations/panoptic_test.json" ,instances_json="dataset_test/annotations/instances_test.json")


#cfg = get_cfg()
#cfg.merge_from_file("../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#cfg.merge_from_file("configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
#cfg.merge_from_file("../configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

cfg = get_cfg()
# for poly lr schedule
add_deeplab_config(cfg)
add_mask_former_config(cfg)
#cfg.merge_from_file("configs/coco-panoptic/swin/maskformer_panoptic_swin_large_IN21k_384_bs64_554k.yaml")
cfg.merge_from_file("configs/ade20k-150/swin/maskformer_swin_tiny_bs16_160k.yaml")
cfg.MODEL.WEIGHTS = "models/" + modelName +".pth"
#cfg.SOLVER.MAX_ITER = 1000*10*5
#cfg.BASE_LR = 0.0002
#cfg.SOLVER.IMS_PER_BATCH = 16
#cfg.DATASETS.TRAIN = ("coco_2017_train_panoptic")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 13
cfg.DATASETS.TRAIN = ("ffi_train_stuffonly", )
cfg.DATASETS.TEST = ("ffi_val_stuffonly", )
cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.5
cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.01

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

cfg.freeze()
#default_setup(cfg, args)
# Setup logger for "mask_former" module
#setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")



register_coco_panoptic_separated(name="ffi_train", metadata={}, sem_seg_root="../../../data/dataset/train/panoptic_stuff_train", image_root="../../../data/dataset/train/images", panoptic_root="../../../data/dataset/train/panoptic_train" , panoptic_json="../../../data/dataset/train/annotations/panoptic_train.json" ,instances_json="../../../data/dataset/train/annotations/instances_train.json")
register_coco_panoptic_separated(name="ffi_val", metadata={}, sem_seg_root="../../../data/dataset/train/panoptic_stuff_train", image_root="../../../data/dataset/train/images", panoptic_root="../../../data/dataset/train/panoptic_train" , panoptic_json="../../../data/dataset/train/annotations/panoptic_train.json" ,instances_json="../../../data/dataset/train/annotations/instances_train.json")


MetadataCatalog.get("ffi_train_stuffonly").set(stuff_classes=['Person', 'Sky', 'Dirtroad', 'Vehicle', 'Forrest', 'CameraEdge', 'Bush', 'Puddle', 'Large_stone', 'Grass', 'Gravel', 'Building'])
MetadataCatalog.get("ffi_train_stuffonly").set(stuff_colors=[[255, 38, 38], [15, 171, 255], [191, 140, 0], [150, 0, 191], [46, 153, 0], [70, 70, 70], [232, 227, 81], [255, 179, 0], [200, 200, 200], [64, 255, 38], [180, 180, 180], [255, 20, 20]])
MetadataCatalog.get("ffi_train_stuffonly").set(stuff_dataset_id_to_contiguous_id={1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12})

MetadataCatalog.get("ffi_val_stuffonly").set(stuff_classes=['Person', 'Sky', 'Dirtroad', 'Vehicle', 'Forrest', 'CameraEdge', 'Bush', 'Puddle', 'Large_stone', 'Grass', 'Gravel', 'Building'])
MetadataCatalog.get("ffi_val_stuffonly").set(stuff_colors=[[255, 38, 38], [15, 171, 255], [191, 140, 0], [150, 0, 191], [46, 153, 0], [70, 70, 70], [232, 227, 81], [255, 179, 0], [200, 200, 200], [64, 255, 38], [180, 180, 180], [255, 20, 20]])
MetadataCatalog.get("ffi_val_stuffonly").set(stuff_dataset_id_to_contiguous_id={1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12})

#cfg.merge_from_list(['MODEL.DEVICE', 'cpu', 'MODEL.WEIGHTS', 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'])
#cfg.merge_from_list(['MODEL.DEVICE', 'cpu', 'MODEL.WEIGHTS', 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl'])
#cfg.merge_from_list(['MODEL.DEVICE', 'cuda', 'MODEL.WEIGHTS', 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl'])
#cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])
#cfg.MODEL.WEIGHTS = "../training/outputMultiGPU/model_final.pth"
#cfg.merge_from_list(['MODEL.DEVICE', 'cpu', 'MODEL.WEIGHTS', 'detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl'])
#https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl

#cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
#cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6 
#cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 8
#cfg.DATASETS.TEST = ("ffi_test_separated", )
#cfg.freeze()
modelYo = build_model(cfg)
#print("aasd1")
#print(modelYo)
#print("aasd2")
#print(cfg)

metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
print(metadata)
#print(len(metadata.thing_classes))
#print(metadata.thing_classes)
#print(metadata.thing_dataset_id_to_contiguous_id)

#width = 1920
#height = 1080
#cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


WINDOW_NAME = "WebCamTest"

video_visualizer = VideoVisualizer(metadata, ColorMode.IMAGE)
predictor = DefaultPredictor(cfg)


#typeOfFrame = "Video"
typeOfFrame = "Image"
#typeOfFrame = "VideoOptak"

index = 0

#startPath = "/home/potetsos/skule/2021Host/prosjektTesting/combined/detectron2/demo/bilder"
#startPath = "/lhome/asbjotof/work/project/training/dataset_test/images"
#startPath = "../../../data/combinedImages/road_drive2"
startPath = "../../../data/dataset/train/images"
startPath = "inputImages2"

files = listdir(startPath)
files.sort()
#print(files)

#typeOfAnalytics = "panoptic"
typeOfAnalytics = "both"

saving = True

print(cfg)

if (typeOfFrame == "Video"):
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
elif (typeOfFrame == "VideoOptak"):
    #cam = cv2.VideoCapture("/home/potetsos/skule/2021Host/prosjektTesting/combined/detectron2/demo/videoer/Kia-eNiro-AV_LowRes.mp4")
    cam = cv2.VideoCapture("/home/potetsos/skule/2021Host/prosjekt/dataFFI/ffiBilder/filmer/rockQuarryIntoWoodsDrive.mp4")
    #cam = cv2.VideoCapture("/home/potetsos/skule/2021Host/prosjekt/dataFFI/ffiBilder/stille.mp4")
    num_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    #codec, file_ext = ("x264", ".mkv")
    codec, file_ext = ("mp4v", ".mp4")
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fname = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/ffiBilder/rockQuarryIntoWoodsDrive_Analyzed" + file_ext
    frames_per_second = cam.get(cv2.CAP_PROP_FPS)
    print("FPS", frames_per_second)
    output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    #frameSize=(3840, 1080),
                    frameSize=(4096, 1536),
                    #frameSize=(2048, 540),
                    isColor=True,
            )
    print("\n")
    print(output_file)
    print("\n")

def getFrame(index):

    #print("index", index)
    if (typeOfFrame == "Video" or typeOfFrame == "VideoOptak"):
        success, frame = cam.read()
        #print(cam.get(cv2.CAP_PROP_POS_MSEC))
        #print("heui")
        #print(cam)
        #print(success)
        #print(frame.shape)   
        #return(frame)
    if (typeOfFrame == "Image"):
        print(str(index) + "/" +str(len(files)))

        frame = read_image(startPath + "/" +files[index], format="BGR")
        #print(frame.shape)
        index+=1


def visualise_predicted_frame(frame, predictions):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_visualizer = Visualizer(frame, metadata)
    sem_seg = predictions["sem_seg"]
    frame_visualizer.draw_sem_seg(sem_seg.argmax(dim=0).to('cpu'), area_threshold=None, alpha=0.5)
    vis_frame = frame_visualizer.output
    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
    return(vis_frame)

def visEachClass(frame, predictions):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_visualizer = Visualizer(frame, metadata)
    sem_seg = predictions["sem_seg"]
    #print(sem_seg)
    #print(sem_seg.shape)
    #print(type(sem_seg[0]))
    #pred = (sem_seg[0] * 128).to('cpu').numpy()
    #print(pred)
    #print(len(sem_seg))

    #print(sem_seg[0])
    #print(sem_seg[5])
    #print(sem_seg[10])

    for classNumber in range(len(sem_seg)):
        #print(classNumber)
        #print(sem_seg[classNumber])
        #print("\n")


        pred = (sem_seg[classNumber] * 128).to('cpu').numpy()
        cv2.imwrite("outputImages/LiDAR/" + modelName + "/" + str(counter) + "/" + str(classNumber) + ".png", pred)


counter=0
totalTime = 0
for fileName in tqdm(files):
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
    visEachClass(frame, predictedPanoptic)
    combinedFrame = np.hstack((vis_panoptic, frame))
    #print(combinedFrame)
    #print(combinedFrame.shape)
    if (saving == True):
        cv2.imwrite("outputImages/LiDAR/" + modelName + "/" + fileName, combinedFrame)

