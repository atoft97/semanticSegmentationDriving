from distutils.command.build import build
from tkinter.messagebox import NO
from LiDARsegmentation.maskFormer.inferance.inferance import Inferance
from tqdm import tqdm
from os import listdir
from detectron2.data.detection_utils import read_image
from driving.classesToDrivable import ClassesToDrivable
import cv2
from driving.segmented2DIrection import Segmented2Direction
from driving.segmented2DIrectionLocal import Segmented2DirectionLocal
import time
#from monoDepth.esimateDepth import EsimateDepth
import rospy
from sensor_msgs.msg import PointCloud2, Image, NavSatFix, Imu, PointField
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from cv_bridge import CvBridge
import numpy as np

from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math

#from ouster import client
import sys

import open3d as o3d
from ouster import client

import ros_numpy
from std_msgs.msg import Header
import copy
import tf

import sensor_msgs.point_cloud2 as pc2
import tf2_ros

import tf.transformations as t

from rospy.msg import AnyMsg
from ouster.client._client import ScanBatcher
import sensor_msgs.msg as sensor_msgs
from cv_bridge import CvBridge
import std_msgs.msg as std_msgs
from geometry_msgs.msg import Twist
#from PIL import Image

class NodeTwistOutput:

    def __init__(self):
        print("initializing NodeTwistOutput")
        #self.pub_twist = rospy.Publisher("~cmd_vel", Twist, queue_size = 1)
        self.sub_lidarSections = rospy.Subscriber("/custum/outputDirection", Twist, self.processTwist ,queue_size=1, buff_size=2*30)
        self.timerLidar = rospy.Timer(rospy.Duration(1./100), self.publishTwist)
        self.pub_twist= rospy.Publisher("/twistOut", Twist, queue_size = 5)
        self.newsetTwist = None
        print("initialized NodeTwistOutput")


    def processTwist(self, inputTwist):
        self.newsetTwist = inputTwist

        degrees = math.degrees(self.newsetTwist.angular.z)

        if (abs(degrees) > 20):
            self.newsetTwist.linear.x = 0
        else:
            self.newsetTwist.angular.z = self.newsetTwist.angular.z * 2
        
        print(self.newsetTwist)
    
    def publishTwist(self, event=None):
        if (self.newsetTwist == None):
            return
        self.pub_twist.publish(self.newsetTwist)



if __name__ == '__main__':
    rospy.init_node('NodeTwistOutput')

    node = NodeTwistOutput() 

    while not rospy.is_shutdown():
        rospy.spin()

    rospy.on_shutdown(node.on_shutdown)
