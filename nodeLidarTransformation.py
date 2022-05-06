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

from ouster_ros.msg import PacketMsg

from rospy.msg import AnyMsg
from ouster.client._client import ScanBatcher
import sensor_msgs.msg as sensor_msgs
from cv_bridge import CvBridge
import std_msgs.msg as std_msgs
#from PIL import Image

class NodeLidarTransform:

    def __init__(self):
        print("initializing lidarTransformNode")

        metadata_path = "lidar_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadataLidar = client.SensorInfo(f.read())
        
        self.ls = client.LidarScan(self.metadataLidar.format.pixels_per_column,
                          self.metadataLidar.format.columns_per_frame)
        self.batch = ScanBatcher(self.metadataLidar)

        self.sub_lidarSections = rospy.Subscriber("/ugv_sensors/lidar/driver/lidar_packets", PacketMsg, self.processLidarSection ,queue_size=1000, buff_size=2*30)

        self.counter = 0
        self.frame_id = 0
        self.segmentID = 0

        self.bridge = CvBridge()

        self.pub_custom_pointcloud = rospy.Publisher("/custom/Pointcloud", PointCloud2, queue_size = 5)
        self.pub_custom_rangeImage = rospy.Publisher("/custom/rangeImage", Image, queue_size = 5)
        self.pub_custom_rangeReflect= rospy.Publisher("/custom/reflectImage", Image, queue_size = 5)
        self.pub_custom_rangeSignal= rospy.Publisher("/custom/signalImage", Image, queue_size = 5)
        
        print("initialized lidarTransformNode")




    def processLidarSection(self, lidarPacket):
        self.counter += 1
        lidarMelding = client.LidarPacket(lidarPacket.buf, self.metadataLidar)
        self.batch(lidarMelding._data, self.ls)

        if (self.ls.frame_id != self.frame_id):
            self.frame_id = self.ls.frame_id
            ranges = self.ls.field(client.ChanField.RANGE)
            refect = self.ls.field(client.ChanField.REFLECTIVITY)
            signal = self.ls.field(client.ChanField.SIGNAL)

            #print(signal)

            range_img = client.destagger(self.metadataLidar, ranges)
            refect_img = client.destagger(self.metadataLidar, refect)
            signal_img = client.destagger(self.metadataLidar, signal)

            print(self.ls.frame_id)
            #cv2.imwrite(f"heimalageBilde/range/{self.ls.frame_id}.png", range_img/255)
            #cv2.imwrite(f"heimalageBilde/refect/{self.ls.frame_id}.png", refect_img/1.0)
            #cv2.imwrite(f"heimalageBilde/signal/{self.ls.frame_id}.png", signal_img/1.0)
            #ls = client.LidarScan(info.format.pixels_per_column,
            #              info.format.columns_per_frame)

            xyzlut = client.XYZLut(self.metadataLidar)
            xyz = xyzlut(self.ls.field(client.ChanField.RANGE))

            pc2 = self.point_cloud(xyz, "os_sensor")
            #print(pc2)

            #print(type(xyz))
            #print(xyz.shape)

            #self.pub_custom_image
            #self.bridge.cv2_to_imgmsg(range_img)
            range_img = (range_img/255).astype('uint8')
            refect_img = (refect_img).astype('uint8')
            signal_img = (signal_img).astype('uint8')
            self.pub_custom_rangeImage.publish(self.bridge.cv2_to_imgmsg(range_img))
            self.pub_custom_rangeReflect.publish(self.bridge.cv2_to_imgmsg(refect_img))
            self.pub_custom_rangeSignal.publish(self.bridge.cv2_to_imgmsg(signal_img))
            self.pub_custom_pointcloud.publish(pc2)
            #xyz = client.XYZLut(metadata)(scan)

            #cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.reshape((-1, 3))))
    def point_cloud(self, points, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx7 array of xyz positions (m) and rgba colors (0..1)
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        """
        ros_dtype = sensor_msgs.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        data = points.astype(dtype).tobytes()

        fields = [sensor_msgs.PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]

        header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())

        return sensor_msgs.PointCloud2(
            header=header,
            height=points.shape[1],
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 3),
            row_step=(itemsize * 3 * points.shape[0]),
            data=data
        )
        #ranges = self.ls.field(client.ChanField.RANGE)
        #print(ranges.shape)
        #range_img = client.destagger(self.metadataLidar, ranges)
        #print(range_img.shape)
        #print(range_img/255)
        #cv2.imwrite(f"heimalageBilde/range/{self.counter}.png", range_img/255)

        #print(self.ls.frame_id)

        #print("\n")

if __name__ == '__main__':
    rospy.init_node('lidarTransfromNode')

    node = NodeLidarTransform() 

    while not rospy.is_shutdown():
        rospy.spin()

    rospy.on_shutdown(node.on_shutdown)
