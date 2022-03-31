import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

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

device = torch.device("cuda")

#register_coco_panoptic_separated(name="ffi_test", sem_seg_root="dataset_test/panoptic_stuff_test", metadata={}, image_root="dataset_test/images", panoptic_root="dataset_test/panoptic_test" , panoptic_json="dataset_test/annotations/panoptic_test.json" ,instances_json="dataset_test/annotations/instances_test.json")

cfg = get_cfg()
add_deeplab_config(cfg)
add_mask_former_config(cfg)
#cfg.merge_from_file("configs/ade20k-150/swin/maskformer_swin_large_IN21k_384_bs16_160k_res640.yaml")
#cfg.MODEL.WEIGHTS = "models/model_final_aefa3b.pkl"

cfg.merge_from_file("configs/ade20k-150/swin/maskformer_swin_tiny_bs16_160k.yaml")
cfg.MODEL.WEIGHTS = "models/model_final_8657a5.pkl"

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5

cfg.freeze()

modelYo = build_model(cfg)


metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
print(metadata)
#print(fail)

WINDOW_NAME = "WebCamTest"

video_visualizer = VideoVisualizer(metadata, ColorMode.IMAGE)
predictor = DefaultPredictor(cfg)

#typeOfFrame = "Video"
typeOfFrame = "Image"
#typeOfFrame = "VideoOptak"

index = 0
#startPath = "../../../data/combinedImagesTaller/plains_drive"
startPath = "../../../data/dataset/train/images"

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

    print(sem_seg[2])
    print(sem_seg[5])
    print(sem_seg[12])

    os.makedirs("outputImages/LiDAR/" + modelName + "/" + str(counter), exist_ok=True)

    for classNumber in range(len(sem_seg)):
        pred = (sem_seg[classNumber] * 128).to('cpu').numpy()
        cv2.imwrite("outputImages/LiDAR/" + modelName + "/" + str(counter) + "/" + str(classNumber) + ".png", pred)


modelName = "semanticStandardTest2"
os.makedirs("outputImages/LiDAR/" + modelName, exist_ok=True)

counter=0
totalTime = 0
for fileName in files:
    counter += 1
    print(str(counter) + "/" +str(len(files)))
    frame = read_image(startPath + "/" +fileName, format="BGR")
    startTime = time.time()
    predictedPanoptic = predictor(frame)
    #print(predictedPanoptic)
    diffTime = time.time() - startTime
    totalTime += diffTime
    print("avgTime", totalTime/counter)
    vis_panoptic = visualise_predicted_frame(frame, predictedPanoptic)
    visEachClass(frame, predictedPanoptic)
    combinedFrame = np.vstack((vis_panoptic, frame))
    if (saving == True):
        cv2.imwrite("outputImages/LiDAR/" + modelName + "/" + fileName, combinedFrame)