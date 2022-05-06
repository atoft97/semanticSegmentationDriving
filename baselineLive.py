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
#from PIL import Image

class BaselineLive:

    def __init__(self, modelName, useDepth, modelType,  goalX, goalY):
        print("initializing")

        metadata_path = "lidar_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadataLidar = client.SensorInfo(f.read())

        self.inferance = Inferance(loggingFolder="", modelName=modelName)
        self.classToDrivable = ClassesToDrivable()
        #self.segmented2Direction = Segmented2Direction("rgb", csvPath="outputDataset/rgbDriving.csv", loggingFolder="")
        #self.segmented2Direction = Segmented2DirectionLocal("rgb", csvPath="outputDataset/rgbDriving.csv", loggingFolder="")
        self.segmented2Direction = Segmented2DirectionLocal(modelType, csvPath="outputDataset/rgbDriving.csv", loggingFolder="", goalX=goalX, goalY=goalY)
        self.goalX = goalX
        self.goalY = goalY
        
        #self.esimateDepth = EsimateDepth()

        #self.latest_longitude = 0
        #self.latest_latitude = 0 
        #self.latets_heading = 0
        
        self.latest_xpos = 0
        self.latest_ypos = 0 
        self.latest_zpos = 0
        self.latets_heading = 0
        self.latest_image = None
        self.last_processed_image = None
        self.latest_reflec = None
        self.latest_range = None
        self.latest_signal = None
        self.latest_pointCloud = None
        self.combinedPointCLoud = o3d.geometry.PointCloud()

        self.latest_rosOdom = None
        self.lindarSyncedOdom = None

        self.lidarSyncedTrans = None
        self.lidarSyncedRot = None

        self.pythonLidarTransform = False
        
        

        #self.sub_cloud = rospy.Subscriber("/ugv_sensors/lidar/cloud/points", PointCloud2, self.processLidarImage,queue_size=1) #konverter sjølv med ouster eller ta inn 4 topics (/ugv_sensors/lidar/image/range_image)
        
        stortTall = 2**32
        #print("stor", stortTall)

        if (modelType=="rgb"):
            self.sub_rgb = rospy.Subscriber("/ugv_sensors/camera/color/image", Image, self.processRGBImage,queue_size=1, buff_size=stortTall)
        elif (modelType=="lidar"):
            self.sub_range = rospy.Subscriber("/ugv_sensors/lidar/image/range_image", Image, self.processRange,queue_size=1, buff_size=stortTall)
            self.sub_signal = rospy.Subscriber("/ugv_sensors/lidar/image/signal_image", Image, self.processSignal,queue_size=1, buff_size=stortTall)
            self.sub_reflec = rospy.Subscriber("/ugv_sensors/lidar/image/reflec_image", Image, self.processReflec,queue_size=1, buff_size=stortTall)
            self.sub_lidar = rospy.Subscriber("/ugv_sensors/lidar/cloud/points", PointCloud2, self.processLidar,queue_size=1, buff_size=stortTall)
            #self.sub_lidar_imu = rospy.Subscriber("/ugv_sensors/lidar/cloud/imu", Imu, self.processLidarImu,queue_size=1, buff_size=stortTall)
            #self.sub_range = rospy.Subscriber("/custom/rangeImage", Image, self.processRange,queue_size=1, buff_size=stortTall)
            #self.sub_signal = rospy.Subscriber("/custom/signalImage", Image, self.processSignal,queue_size=1, buff_size=stortTall)
            #self.sub_reflec = rospy.Subscriber("/custom/reflectImage", Image, self.processReflec,queue_size=1, buff_size=stortTall)
            #self.sub_lidar = rospy.Subscriber("/custom/Pointcloud", PointCloud2, self.processLidar,queue_size=1, buff_size=stortTall)
            
            

        #self.sub_lidar = rospy.Subscriber("/ugv_sensors/lidar/cloud/points", PointCloud2, self.processLidar,queue_size=1, buff_size=stortTall)
        #self.sub_pos = rospy.Subscriber("/ugv_sensors/navp_ros/nav_sat_fix", NavSatFix, self.processPOS,queue_size=1)
        #self.sub_pos = rospy.Subscriber("/warpath/navigation/odometry_integrated_center_enu", NavSatFix, self.processPOS,queue_size=1)
        #self.sub_heading = rospy.Subscriber("/warpath/navigation/odometry_integrated_center_enu", Odometry, self.processHeading,queue_size=1)
        self.sub_pos_heading = rospy.Subscriber("/warpath/navigation/odometry_integrated_center_enu", Odometry, self.processHeadingAndPos,queue_size=1)
        #self.sub_heading = rospy.Subscriber("/ugv_sensors/navp_ros/navp_msg", Odometry, self.processHeading,queue_size=1)
        self.bridge = CvBridge()

        self.pub_twist = rospy.Publisher("/custum/outputDirection", Twist, queue_size = 1)
        self.pub_segmented_image = rospy.Publisher("~segmentedImage", Image, queue_size = 1)
        self.pub_segmented_pointcloud = rospy.Publisher("~segmentedPointCloud", PointCloud2, queue_size = 1)

        #self.freq = rospy.get_param("~freq", 5.)
        #print("freq", self.freq)
        if (modelType=="rgb"):
            self.timer = rospy.Timer(rospy.Duration(1./10), self.timerFunction)
        elif (modelType=="lidar"):
            self.timerLidar = rospy.Timer(rospy.Duration(1./10), self.timerLidar)
            #self.timeLidarPointCloud = rospy.Timer(rospy.Duration(1./10), self.timeLidarPointCloud)
        #self.timer = rospy.Timer(rospy.Duration(1./10), self.timerFunction)
        
        #print(self.timer)

        #metadata_path = "lidar_metadata.json"

        #with open(metadata_path, 'r') as f:
        #    metadata = client.SensorInfo(f.read())
        #self.tfBuffer = tf2_ros.Buffer()
        #self.listener = tf.TransformListener(self.tfBuffer)

        print("initialized")
        self.imageCounter = 0

        self.ready = True

        self.imageCounter = 0

    #def processLidarImu(self, rosLidarImu):
    #    print(rosLidarImu)

    def processRange(self, rosImage):
        #print("range:", rosImage.header.seq)
        #print("new range")
        self.latest_range = rosImage
    
    def processSignal(self, rosImage):
        #print("Signal:", rosImage.header.seq)
        self.latest_signal = rosImage
    
    def processReflec(self, rosImage):
        #print("Reflec:", rosImage.header.seq)
        self.latest_reflec = rosImage

    def processLidar(self, rosPointcloud):
        #print("lidar:", rosPointcloud.header.seq)
        self.latest_pointCloud = rosPointcloud
        self.lindarSyncedOdom = copy.deepcopy(self.latest_rosOdom)
        
        '''
        try:
            
            #(trans,rot) = self.listener.lookupTransform('/Qe', '/B', rospy.Time(0))
            (trans,rot) = self.listener.lookupTransform('/B', '/Qe', rospy.Time(0))
            self.lidarSyncedTrans = trans
            self.lidarSyncedRot = rot
            print("\n\n\n\n\n\n")
            print("trans rot", trans, rot)
            print("odom", self.latest_rosOdom)
            print("\n\n\n\n\n\n")
            
            (trans,rot) = self.listener.lookupTransform('/Qe', '/Bcre', rospy.Time(0))
            print("\n\n\n\n\n\n")
            print("lf trans rot", trans, rot)
            print("odom", self.latest_rosOdom)
            print("\n\n\n\n\n\n")
            
            #(trans,rot) = self.listener.lookupTransform('/os_sensor', '/os1_lidar', rospy.Time(0))
            #print("\n\n\n\n\n\n")
            #print("lf trans rot", trans, rot)
            #print("odom", self.latest_rosOdom)
            #print("\n\n\n\n\n\n")
            
            #(trans,rot) = self.listener.lookupTransform('/os_sensor', '/Qe', rospy.Time(0))
            (trans,rot) = self.listener.lookupTransform('/Qe', '/os_sensor', rospy.Time(0))
            self.lidarSyncedTrans = trans
            self.lidarSyncedRot = rot
            #print("\n\n\n\n\n\n")
            #print("lf trans rot", trans, rot)
            #print("odom", self.latest_rosOdom)
            #print("\n\n\n\n\n\n")
        except:
            print("Process lidar: Ingen TF")
        '''
        


    
    def processHeadingAndPos(self, rosOdom):
        self.latest_rosOdom = rosOdom

        #print(rosOdom.header)
        self.latest_xpos = rosOdom.pose.pose.position.x
        self.latest_ypos = rosOdom.pose.pose.position.y
        self.latest_zpos = rosOdom.pose.pose.position.z
        #fix heading
        #([0.06146124, 0, 0, 0.99810947])
        #print(rosOdom.pose.pose.orientation)
        orientationlist = (rosOdom.pose.pose.orientation.x, rosOdom.pose.pose.orientation.y, rosOdom.pose.pose.orientation.z, rosOdom.pose.pose.orientation.w)
        
        euler = euler_from_quaternion(orientationlist)
        #print(euler)
        headingRad = euler[2]
        heading = math.degrees(headingRad)
        #print(heading)
        self.latets_heading = heading
    
    def processPOS(self, rosPos):
        #print(rosPos)
        self.latest_longitude = rosPos.longitude
        self.latest_latitude = rosPos.latitude

        
        
    def processHeading(self, data):
        #print(type(data))
        #print(data)
        #print(dir(data))
        #connection_header = data._connection_header['type'].split('/')
        #ros_pkg = connection_header[0] + '.msg'
        #msg_type = connection_header[1]
        #print(msg_type)
        #print(ros_pkg)
        #print(data._full_text)
        #print("etter full")
        heading = 200
        self.latets_heading = heading
        #msg_class = getattr(import_module(ros_pkg), msg_type)

    def timerFunction(self, event=None):
        #print("timer")
        rosImage = self.latest_image
        odom = copy.deepcopy(self.lindarSyncedOdom)

        if (rosImage == None or odom == None):
            #print("tomt bilde")
            return

        rosY = odom.pose.pose.position.y
        rosX = odom.pose.pose.position.x
        rosZ = odom.pose.pose.position.z

        rosRotationX = odom.pose.pose.orientation.x
        rosRotationY = odom.pose.pose.orientation.y
        rosRotationZ = odom.pose.pose.orientation.z
        rosRotationW = odom.pose.pose.orientation.w

        
        #print(type(rosImage))
        #print(rosImage)
        #if (np.array_equal(rosImage, self.last_processed_image)):
        #    return
        odom = copy.copy(self.lindarSyncedOdom)

        if (self.goalX -2 < rosY < self.goalX + 2 and self.goalY -2 < rosX < self.goalY + 2):
            twist = Twist()
            twist.linear.x = 0 #constant low speed for testing
            twist.angular.z = 0 # mabye change to negative for counter clockwise
            self.pub_twist.publish(twist)
            print("goal reached")
            return



        if (self.ready):
        #    self.last_processed_image = rosImage

            self.ready = False

            self.imageCounter +=1 
            lokalCounter = self.imageCounter
            print("start", lokalCounter)
            
            #if (self.ready):
            #self.ready = False

            fileName = "tmp.png"
            loggingFolder = ""
            useDepth = False
            #if ferdigprossesert, gå videre
            #print("bidle motatt")
            #print(type(rosImage))
            image = self.bridge.imgmsg_to_cv2(rosImage, desired_encoding="bgr8")
            
            #print(type(image))

            vis_panoptic, rgb_img, classImage = self.inferance.segmentImage(image, fileName)
            cv2.imwrite("testRGB.png", rgb_img)

            if (useDepth):
                depthImage = self.esimateDepth.midasPredict(image) #add stero vision
            else:
                depthImage=False

            #cv2.imwrite("outputData/rgbImages/test" +str(self.imageCounter)+".png", rgb_img)
            drivableIndex, drivableColor = self.classToDrivable.imageClassesToDrivable(rgb_img, fileName, loggingFolder)


            eulerOdom = euler_from_quaternion((rosRotationX, rosRotationY, rosRotationZ, rosRotationW))
            xrotationEtter = 180-math.degrees(eulerOdom[2])
            print("roatasjon", xrotationEtter)

            print("pos", rosX, rosY)

            #recomendedDirection = self.segmented2Direction.getDirectionOfImage(drivableIndex, vis_panoptic, fileName, combinedTaller[:,:,2], rgb_img, drivableColor, useDepth=True, lat=rosX, long=rosY, heading=xrotationEtter)
            #print(recomendedDirection)

            recomendedDirection = self.segmented2Direction.getDirectionOfImage(drivableIndex, vis_panoptic, fileName, depthImage, rgb_img, drivableColor, useDepth=useDepth, lat=rosX, long=rosY, heading=xrotationEtter)
            print(recomendedDirection)

            #send twist            
            twist = Twist()
            twist.linear.x = 1 #constant low speed for testing
            twist.angular.z = -math.radians(recomendedDirection) # mabye change to negative for counter clockwise
            self.pub_twist.publish(twist)

            rgb_img = rgb_img.astype('uint8')
            print("publisert bilde")
            #self.pub_segmented_image.publish(self.bridge.cv2_to_imgmsg(rgb_img))


            

            #self.ready = True

            print("end", lokalCounter)
            print("\n")
            self.ready = True

    
    def timeLidarPointCloud(self, event=None):
        print("time lidar")
        
        rosReflec = copy.copy(self.latest_reflec)
        rosRange = copy.copy(self.latest_range)
        rosSignal= copy.copy(self.latest_signal) 
        rosPointCLoud = copy.copy(self.latest_pointCloud)
        #rosY = copy.copy(self.latest_ypos)
        #rosX = copy.copy(self.latest_xpos)
        #rosZ = copy.copy(self.latest_zpos)

        odom = copy.copy(self.lindarSyncedOdom)

        

        

        # or self.lidarSyncedTrans==None
        if (rosRange == None or rosReflec==None or rosSignal==None or rosPointCLoud==None or odom==None or self.lidarSyncedTrans==None):
            #print("tomt bilde")
            print("time lidar: Ingen TF")
            return



        rosY = odom.pose.pose.position.y
        rosX = odom.pose.pose.position.x
        rosZ = odom.pose.pose.position.z

        rosRotationX = odom.pose.pose.orientation.x
        rosRotationY = odom.pose.pose.orientation.y
        rosRotationZ = odom.pose.pose.orientation.z
        rosRotationW = odom.pose.pose.orientation.w

        if (self.goalX -2 < rosY < self.goalX + 2 and self.goalY -2 < rosX < self.goalY + 2):
            twist = Twist()
            twist.linear.x = 0 #constant low speed for testing
            twist.angular.z = 0 # mabye change to negative for counter clockwise
            self.pub_twist.publish(twist)
            print("goal reached")
            return


        print(rosReflec.header)
        print(rosRange.header)
        print(rosSignal.header)
        print(rosPointCLoud.header)
        print(odom.header)
        #print(self.lidarSyncedTrans)

        if (self.ready):
        #    self.last_processed_image = rosImage

            self.ready = False

            self.imageCounter +=1 
            lokalCounter = self.imageCounter
            print("start", lokalCounter)
            
            #if (self.ready):
            #self.ready = False

            fileName = "tmp.png"
            loggingFolder = ""
            useDepth = False
            #if ferdigprossesert, gå videre
            #print("bidle motatt")
            #print(type(rosImage))
            #image = self.bridge.imgmsg_to_cv2(rosImage, desired_encoding="bgr8")

            rangeImage = self.bridge.imgmsg_to_cv2(rosRange, desired_encoding="bgr8")
            reflecImage = self.bridge.imgmsg_to_cv2(rosReflec, desired_encoding="bgr8")
            signalImage = self.bridge.imgmsg_to_cv2(rosSignal, desired_encoding="bgr8")

            rangeImage = cv2.cvtColor(rangeImage, cv2.COLOR_BGR2GRAY)
            reflecImage = cv2.cvtColor(reflecImage, cv2.COLOR_BGR2GRAY)
            signalImage = cv2.cvtColor(signalImage, cv2.COLOR_BGR2GRAY)

            rangeImageBright = cv2.multiply(rangeImage, 15) #multply by 15
            rangeImageBright[rangeImageBright <= 0] = 255 #caps at 255
            rangeImageBright = cv2.bitwise_not(rangeImageBright)
            
            #print(type(image))
            #print(rangeImage.shape)
            #print(rangeImage)
            combined = cv2.merge((signalImage, reflecImage, rangeImageBright))
            cv2.imwrite("testCombined.png", combined)

            width = combined.shape[1] # keep original width
            height = 64*4
            dim = (width, height)
            combinedTaller = cv2.resize(combined, dim, interpolation = cv2.INTER_AREA)

            
            vis_panoptic, rgb_img, classImage = self.inferance.segmentImage(combinedTaller, fileName)
            cv2.imwrite("testLidar.png", rgb_img)
            
            #if (useDepth):
            #    depthImage = self.esimateDepth.midasPredict(image) #add stero vision
            #else:
            #    depthImage=False

            #cv2.imwrite("outputData/rgbImages/test" +str(self.imageCounter)+".png", rgb_img)
            drivableIndex, drivableColor = self.classToDrivable.imageClassesToDrivable(rgb_img, fileName, loggingFolder)

            #(drivableIndex, vis_panoptic, fileName, frame[:, :, 2], rgb_img, drivableColor, useDepth = True)

            eulerOdom = euler_from_quaternion((rosRotationX, rosRotationY, rosRotationZ, rosRotationW))
            xrotationEtter = 180-math.degrees(eulerOdom[2])
            print("roatasjon", xrotationEtter)

            print("pos", rosX, rosY)

            recomendedDirection = self.segmented2Direction.getDirectionOfImage(drivableIndex, vis_panoptic, fileName, combinedTaller[:,:,2], rgb_img, drivableColor, useDepth=True, lat=rosX, long=rosY, heading=xrotationEtter)
            print(recomendedDirection)


            #rgb bilde mindre
            width = rgb_img.shape[1] # keep original width
            height = 64
            dim = (width, height)

            rgbSmaller = cv2.resize(rgb_img, dim, interpolation = cv2.INTER_AREA)
            rgbSmaller = cv2.cvtColor(rgbSmaller.astype('float32'), cv2.COLOR_BGR2RGB)
            segmentation_img_staggered = client.destagger(self.metadataLidar, rgbSmaller, inverse=True)
            cv2.imwrite("stagg.png", segmentation_img_staggered)
            segmentation_img_staggeredFloat = (segmentation_img_staggered.astype(float) / 255.0).reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            numpyPointcloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(rosPointCLoud, remove_nans=True)
            #latest_pointCloud
            print(numpyPointcloud.shape)
            print(segmentation_img_staggeredFloat.shape)

            print(numpyPointcloud)
            #numpyPointcloud = numpyPointcloud[:, 2] = 5
            #print(numpyPointcloud[0,2])
            numpyPointcloud[:, 2] = 0
            print(numpyPointcloud)

            pcd.points = o3d.utility.Vector3dVector(numpyPointcloud)
            pcd.colors = o3d.utility.Vector3dVector(segmentation_img_staggeredFloat)
            

            #pcd.translate([-rosY,rosX,rosZ])
            
            #downpcd.translate([10*self.imageCounter,0,0])
            #R = pcd.get_rotation_matrix_from_xyz((0, np.pi-0.3, 0))

            '''
            #os_sensor to base, static origation and strnasform
            R = pcd.get_rotation_matrix_from_quaternion((0.98955, 0.0321437, 0.139623, 0.016217))
            pcd.rotate(R, center=(0,0,0))
            pcd.translate([0.0147109, -0.0294761, -0.298861])

            #base to qe orientation, static orientation
            R_base_qe = pcd.get_rotation_matrix_from_quaternion((-0.701648, 0.712015, -0.0116443, -0.0242871))
            pcd.rotate(R_base_qe, center=(0,0,0))
            '''
            

            #R_odom = pcd.get_rotation_matrix_from_quaternion((rosRotationX, rosRotationY, rosRotationZ, rosRotationW))
            #pcd.rotate(R_odom, center=(0,0,0))
            #pcd.translate([rosX,rosY,rosZ])
            
            
            

            #downpcd.translate([-rosY,rosX,rosZ])

            

            #print(np.asarray(self.combinedPointCLoud.points).shape)


            #roter fra os_sensor til B
            #R_odom = pcd.get_rotation_matrix_from_quaternion((-0.139623, 0.016217, 0.98955, 0.0321437))
            #pcd.rotate(R_odom, center=(0,0,0))
            #pcd.translate([-1.07017, -0.0179944, 0.291813])

            #roter fra B til Qe
            #self.lidarSyncedRot
            print("x and y an z", rosX, rosY, rosZ)
            #R_odom = pcd.get_rotation_matrix_from_quaternion(([ 0, 0, 0.7071068, 0.7071068 ]))
            #pcd.rotate(R_odom, center=(0,0,0))
            #euler = euler_from_quaternion(self.lidarSyncedRot)
            eulerOdom = euler_from_quaternion((rosRotationX, rosRotationY, rosRotationZ, rosRotationW))
            xrotation = math.degrees(eulerOdom[0])
            yrotation = math.degrees(eulerOdom[1])
            zrotation = math.degrees(eulerOdom[2])

            #xrotation = math.degrees(euler[0])
            #yrotation = math.degrees(euler[1])
            #zrotation = math.degrees(euler[2])
            #zrotation += 90

            eulerDegrees = [xrotation, yrotation, zrotation]

            quaternion = quaternion_from_euler(math.radians(xrotation), math.radians(yrotation), math.radians(zrotation))

            #R_odom = pcd.get_rotation_matrix_from_quaternion(quaternion)
            #pcd.rotate(R_odom, center=(0,0,0))

            print(eulerOdom)
            print("kjoretoys rotasjon:", eulerOdom)
            print("kjoretoy degrees:", eulerDegrees)

            #pcd.translate([rosX, rosY, rosZ])
            print("Function: Rotation for: ", self.lidarSyncedRot)
            eulerFor = euler_from_quaternion(self.lidarSyncedRot)
            xrotationFor = math.degrees(eulerFor[0])
            yrotationFor = math.degrees(eulerFor[1])
            zrotationFor = math.degrees(eulerFor[2])
            eulerDegreesFor = [xrotationFor, yrotationFor, zrotationFor]
            print("Function: Rotation for degrees: ", eulerDegreesFor)

            eulerEtter = euler_from_quaternion(self.lidarSyncedRot)
            #xrotationEtter = 180-math.degrees(eulerEtter[2])
            xrotationEtter = 180-math.degrees(eulerOdom[2])
            #yrotationEtter = -math.degrees(eulerEtter[1])
            #zrotationEtter = math.degrees(eulerEtter[1])
            #xrotationEtter = 180+10
            yrotationEtter = 0
            zrotationEtter = 0

            #Function: Rotation for degrees:  [-0.45820136421640933, 10.879432430919548, -10.578782050131228]
            #ca 180 x

            eulerDegreesEtter = [xrotationEtter, yrotationEtter, zrotationEtter]
            print("Function: Rotation etter degrees: ", eulerDegreesEtter)
            quaternion = quaternion_from_euler(math.radians(xrotationEtter), math.radians(yrotationEtter), math.radians(zrotationEtter))
            '''
            opposite = [-self.lidarSyncedRot[0], -self.lidarSyncedRot[1], -self.lidarSyncedRot[2], -self.lidarSyncedRot[3]]

            print("Function: Rotation etter: ", opposite)
            eulerEtter = euler_from_quaternion(opposite)
            xrotationEtter = math.degrees(eulerEtter[0])
            yrotationEtter = math.degrees(eulerEtter[1])
            zrotationEtter = math.degrees(eulerEtter[2])
            eulerDegreesEtter = [xrotationEtter, yrotationEtter, zrotationEtter]
            print("Function: Rotation etter degrees: ", eulerDegreesEtter)
            '''

            print("Function: Trans: ", self.lidarSyncedTrans)
            R_odom = pcd.get_rotation_matrix_from_quaternion(quaternion)
            #R_odom = pcd.get_rotation_matrix_from_quaternion(quaternion)
            #R_odom = pcd.get_rotation_matrix_from_axis_angle(quaternion)
            #R_odom = pcd.get_rotation_matrix_from_quaternion((0.00478658, 0.0947993, -0.091367, 0.991283))
            #R_odom = pcd.get_rotation_matrix_from_quaternion((-0.00476625, -0.0947785, 0.0913871, 0.991283))
            #R_odom = pcd.get_rotation_matrix_from_quaternion((-0.11217, -0.0960606, 0.721003, 0.677012))
            pcd.rotate(R_odom, center=(0,0,0))
            #pcd.translate(self.lidarSyncedTrans)
            pcd.translate([rosX, rosY, rosZ])


            #pcd.translate(self.lidarSyncedTrans)
            
            #pcd.rotate(R_odom)
            
            #pcd.translate([407.527, 64.3689, 20.7503])
            #pcd.translate([-378.12, -137.309, -94.971])
            


            #pcd.translate([72.4152, -405.742, 5.95755])
            
            

            #se3 = t.quaternion_matrix(self.lidarSyncedRot)
            #se3[0:3, -1] = self.lidarSyncedTrans
            #pcd.transform(se3)

            


            
            self.combinedPointCLoud += pcd
            self.combinedPointCLoud = self.combinedPointCLoud.voxel_down_sample(voxel_size=0.3)
            
            if (self.imageCounter == 20):
                o3d.visualization.draw_geometries([self.combinedPointCLoud])


            #send twist            
            twist = Twist()
            twist.linear.x = 1 #constant low speed for testing
            twist.angular.z = -math.radians(recomendedDirection) # mabye change to negative for counter clockwise
            self.pub_twist.publish(twist)

            #outputImage = Image()
            rgb_img = rgb_img.astype('uint8')
            self.pub_segmented_image.publish(self.bridge.cv2_to_imgmsg(rgb_img))
            
            '''
            FIELDS_XYZ = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            FIELDS_XYZRGB = FIELDS_XYZ + [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
            BIT_MOVE_16 = 2**16
            BIT_MOVE_8 = 2**8
            header = rosPointCLoud.header
            points=np.asarray(pcd.points)
            fields=FIELDS_XYZRGB
            colors = np.floor(np.asarray(pcd.colors)*255)
            colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2] 
            cloud_data=np.c_[points, colors]

            rosPontCloud = pc2.create_cloud(header, fields, cloud_data)
            '''
            rosPontCloud = self.o3dpc_to_rospc(pcd, rosPointCLoud.header)
            #self.pub_segmented_pointcloud.publish(rosPontCloud)
            #o3d.visualization.draw_geometries([pcd])

            #outputImage = Image()
            #self.pub_segmented_image.publish(rgb_img)

            #self.ready = True

            print("end", lokalCounter)
            print("\n")
            self.ready = True

    def o3dpc_to_rospc(self, o3dpc, header, frame_id=None, stamp=None):
        """ convert open3d point cloud to ros point cloud
        Args:
            o3dpc (open3d.geometry.PointCloud): open3d point cloud
            frame_id (string): frame id of ros point cloud header
            stamp (rospy.Time): time stamp of ros point cloud header
        Returns:
            rospc (sensor.msg.PointCloud2): ros point cloud message
        """
        BIT_MOVE_16 = 2**16
        BIT_MOVE_8 = 2**8
        cloud_npy = np.asarray(copy.deepcopy(o3dpc.points))
        is_color = o3dpc.colors
            

        n_points = len(cloud_npy[:, 0])
        if is_color:
            data = np.zeros(n_points, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32)
            ])
        else:
            data = np.zeros(n_points, dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32)
                ])
        data['x'] = cloud_npy[:, 0]
        data['y'] = cloud_npy[:, 1]
        data['z'] = cloud_npy[:, 2]
        
        if is_color:
            rgb_npy = np.asarray(copy.deepcopy(o3dpc.colors))
            rgb_npy = np.floor(rgb_npy*255) # nx3 matrix
            rgb_npy = rgb_npy[:, 0] * BIT_MOVE_16 + rgb_npy[:, 1] * BIT_MOVE_8 + rgb_npy[:, 2]  
            rgb_npy = rgb_npy.astype(np.uint32)
            data['rgb'] = rgb_npy

        rospc = ros_numpy.msgify(PointCloud2, data)
        if frame_id is not None:
            rospc.header.frame_id = frame_id

        if stamp is None:
            rospc.header.stamp = rospy.Time.now()
        else:
            rospc.header.stamp = stamp
        rospc.height = 1
        rospc.width = n_points
        rospc.fields = []
        rospc.fields.append(PointField(
                                name="x",
                                offset=0,
                                datatype=PointField.FLOAT32, count=1))
        rospc.fields.append(PointField(
                                name="y",
                                offset=4,
                                datatype=PointField.FLOAT32, count=1))
        rospc.fields.append(PointField(
                                name="z",
                                offset=8,
                                datatype=PointField.FLOAT32, count=1))    

        if is_color:
            rospc.fields.append(PointField(
                            name="rgb",
                            offset=12,
                            datatype=PointField.UINT32, count=1))    
            rospc.point_step = 16
        else:
            rospc.point_step = 12
        
        rospc.is_bigendian = False
        rospc.row_step = rospc.point_step * n_points
        rospc.is_dense = True
        rospc.header = header
        #rospc.header.stamp = rospy.Time.now()
        return rospc

    def timerLidar(self, event=None):

        
        #print("timer")
        #rosImage = self.latest_image

        #rosReflec = self.latest_reflec
        #rosRange = self.latest_range
        #rosSignal= self.latest_signal 

        odom = copy.copy(self.lindarSyncedOdom)

        rosReflec = copy.copy(self.latest_reflec)
        rosRange = copy.copy(self.latest_range)
        rosSignal= copy.copy(self.latest_signal) 


        

        if (rosRange == None or rosReflec==None or rosSignal==None or odom==None):
            #print("tomt bilde")
            return
        

        rosY = odom.pose.pose.position.y
        rosX = odom.pose.pose.position.x
        rosZ = odom.pose.pose.position.z

        rosRotationX = odom.pose.pose.orientation.x
        rosRotationY = odom.pose.pose.orientation.y
        rosRotationZ = odom.pose.pose.orientation.z
        rosRotationW = odom.pose.pose.orientation.w

        if (self.goalX -2 < rosY < self.goalX + 2 and self.goalY -2 < rosX < self.goalY + 2):
            twist = Twist()
            twist.linear.x = 0 #constant low speed for testing
            twist.angular.z = 0 # mabye change to negative for counter clockwise
            self.pub_twist.publish(twist)
            print("goal reached")
            return

        #print(type(rosImage))
        #print(rosImage)
        #if (np.array_equal(rosImage, self.last_processed_image)):
        #    return


        if (self.ready):
        #    self.last_processed_image = rosImage

            self.ready = False

            self.imageCounter +=1 
            lokalCounter = self.imageCounter
            print("start", lokalCounter)
            
            #if (self.ready):
            #self.ready = False

            fileName = "tmp.png"
            loggingFolder = ""
            useDepth = False
            #if ferdigprossesert, gå videre
            #print("bidle motatt")
            #print(type(rosImage))
            #image = self.bridge.imgmsg_to_cv2(rosImage, desired_encoding="bgr8")

            '''
            rangeImage = self.bridge.imgmsg_to_cv2(rosRange)
            reflecImage = self.bridge.imgmsg_to_cv2(rosReflec)
            signalImage = self.bridge.imgmsg_to_cv2(rosSignal)

            #rangeImage = cv2.cvtColor(rangeImage, cv2.COLOR_BGR2GRAY)
            #reflecImage = cv2.cvtColor(reflecImage, cv2.COLOR_BGR2GRAY)
            #signalImage = cv2.cvtColor(signalImage, cv2.COLOR_BGR2GRAY)
            reflecImageBright = cv2.multiply(reflecImage, 3)
            signalImageBright = cv2.multiply(signalImage, 3)
            rangeImageBright = cv2.multiply(rangeImage, 5) #multply by 15
            #print("for", rangeImageBright)
            rangeImageBright[rangeImageBright <= 1] = 255 #caps at 255
            rangeImageBright = cv2.bitwise_not(rangeImageBright)
            '''
            #print(rosRange)
            #cv2.imwrite("range.png", rosRange)
            if (self.pythonLidarTransform):
                rangeImage = self.bridge.imgmsg_to_cv2(rosRange)
                reflecImage = self.bridge.imgmsg_to_cv2(rosReflec)
                signalImage = self.bridge.imgmsg_to_cv2(rosSignal)

                reflecImageBright = cv2.multiply(reflecImage, 3)
                signalImageBright = cv2.multiply(signalImage, 3)
                rangeImageBright = cv2.multiply(rangeImage, 5) #multply by 15

                rangeImageBright[rangeImageBright <= 5] = 255 #caps at 255
                rangeImageBright = cv2.bitwise_not(rangeImageBright)
                combined = cv2.merge((signalImageBright, reflecImageBright, rangeImageBright))
            else:
                rangeImage = self.bridge.imgmsg_to_cv2(rosRange, desired_encoding="bgr8")
                reflecImage = self.bridge.imgmsg_to_cv2(rosReflec, desired_encoding="bgr8")
                signalImage = self.bridge.imgmsg_to_cv2(rosSignal, desired_encoding="bgr8")

                rangeImage = cv2.cvtColor(rangeImage, cv2.COLOR_BGR2GRAY)
                reflecImage = cv2.cvtColor(reflecImage, cv2.COLOR_BGR2GRAY)
                signalImage = cv2.cvtColor(signalImage, cv2.COLOR_BGR2GRAY)

                rangeImageBright = cv2.multiply(rangeImage, 15) #multply by 15
                rangeImageBright[rangeImageBright <= 0] = 255 #caps at 255
                rangeImageBright = cv2.bitwise_not(rangeImageBright)
                combined = cv2.merge((signalImage, reflecImage, rangeImageBright))
            #print("range shape", rosRange.shape)
            #cv2.imwrite("range.png", rosRange ,1)
            
            #rangeImage = cv2.cvtColor(rangeImage, cv2.COLOR_BGR2GRAY)
            #reflecImage = cv2.cvtColor(reflecImage, cv2.COLOR_BGR2GRAY)
            #signalImage = cv2.cvtColor(signalImage, cv2.COLOR_BGR2GRAY)



            #rangeImageBright = cv2.multiply(rangeImage, 15) #multply by 15
            #rangeImageBright[rangeImageBright <= 0] = 255 #caps at 255
            #rangeImageBright = cv2.bitwise_not(rangeImageBright)


            
            #print(type(image))
            #print(rangeImage.shape)
            #print(rangeImage)
            #combined = cv2.merge((signalImage, reflecImage, rangeImageBright))
            
            #cv2.imwrite("testCombined.png", combined)

            width = combined.shape[1] # keep original width
            height = 64*4
            dim = (width, height)
            combinedTaller = cv2.resize(combined, dim, interpolation = cv2.INTER_AREA)

            
            vis_panoptic, rgb_img, classImage = self.inferance.segmentImage(combinedTaller, fileName)
            #cv2.imwrite("testLidar.png", rgb_img)
            
            #if (useDepth):
            #    depthImage = self.esimateDepth.midasPredict(image) #add stero vision
            #else:
            #    depthImage=False

            #cv2.imwrite("outputData/rgbImages/test" +str(self.imageCounter)+".png", rgb_img)
            drivableIndex, drivableColor = self.classToDrivable.imageClassesToDrivable(rgb_img, fileName, loggingFolder)

            eulerOdom = euler_from_quaternion((rosRotationX, rosRotationY, rosRotationZ, rosRotationW))
            xrotationEtter = -math.degrees(eulerOdom[2])+180
            print(rosX, rosY, xrotationEtter)

            #rad_actual = math.radians(self.latets_heading)
            #x_heading = math.cos(rad_actual)
            #y_heading = math.sin(rad_actual)

            #print(x_heading, y_heading)

            #(drivableIndex, vis_panoptic, fileName, frame[:, :, 2], rgb_img, drivableColor, useDepth = True)
            recomendedDirection, globalDirection = self.segmented2Direction.getDirectionOfImage(drivableIndex, vis_panoptic, fileName, combinedTaller[:,:,2], rgb_img, drivableColor, useDepth=True, lat=rosX, long=rosY, heading=xrotationEtter)
            #recomendedDirection = self.segmented2Direction.getDirectionOfImage(drivableIndex, vis_panoptic, fileName, combinedTaller[:,:,2], rgb_img, drivableColor, useDepth=True, lat=self.latest_xpos, long=self.latest_ypos, heading=self.latets_heading)
            print(recomendedDirection)

            #send twist            
            twist = Twist()
            twist.linear.x = 1 #constant low speed for testing
            #twist.angular.z = -math.radians(recomendedDirection)
            twist.angular.z = math.radians(globalDirection)
             # mabye change to negative for counter clockwise
            self.pub_twist.publish(twist)

            #outputImage = Image()
            rgb_img = rgb_img.astype('uint8')
            
            #self.pub_segmented_image.publish(self.bridge.cv2_to_imgmsg(rgb_img))
            

            #outputImage = Image()
            #self.pub_segmented_image.publish(rgb_img)

            #self.ready = True

            print("end", lokalCounter)
            print("\n")
            self.ready = True
            

        
    def processRGBImage(self, rosImage):

        self.latest_image = rosImage
        self.lindarSyncedOdom = copy.deepcopy(self.latest_rosOdom)

        
        #print("\n")
        #print("nytt bilde")
        #print(self.latest_image)
        #print("\n")

        '''

        self.imageCounter +=1 
        lokalCounter = self.imageCounter
        print("start", lokalCounter)
        
        #if (self.ready):
        #self.ready = False

        fileName = "tmp.png"
        loggingFolder = ""
        useDepth = False
        #if ferdigprossesert, gå videre
        #print("bidle motatt")
        #print(type(rosImage))
        image = self.bridge.imgmsg_to_cv2(rosImage, desired_encoding="bgr8")
        
        #print(type(image))

        vis_panoptic, rgb_img, classImage = self.inferance.segmentImage(image, fileName)
        cv2.imwrite("testRGB.png", rgb_img)

        if (useDepth):
            depthImage = self.esimateDepth.midasPredict(image) #add stero vision
        else:
            depthImage=False

        #cv2.imwrite("outputData/rgbImages/test" +str(self.imageCounter)+".png", rgb_img)
        drivableIndex, drivableColor = self.classToDrivable.imageClassesToDrivable(rgb_img, fileName, loggingFolder)


        recomendedDirection = self.segmented2Direction.getDirectionOfImage(drivableIndex, vis_panoptic, fileName, depthImage, rgb_img, drivableColor, useDepth=useDepth, lat=self.latest_xpos, long=self.latest_ypos, heading=self.latets_heading)
        print(recomendedDirection)

        #send twist            
        twist = Twist()
        twist.linear = 5 #constant low speed for testing
        twist.angular = recomendedDirection

        self.pub_twist.publish(twist)

        #outputImage = Image()
        #self.pub_segmented_image.publish(rgb_img)

        #self.ready = True

        print("end", lokalCounter)
        print("\n")
        

    def runRGBBaselineFromFolder(self, folder, modelName, imageStartPath, useDepth=False):
        #self.inferance = Inferance(loggingFolder="lidarSeg", modelName=modelName, imageStartPath=imageStartPath)
        self.inferance = Inferance(loggingFolder="", modelName=modelName, imageStartPath=imageStartPath)
        self.classToDrivable = ClassesToDrivable()
        self.segmented2Direction = Segmented2Direction("rgb", datasetName=folder, csvPath="outputDataset/rgbDriving.csv", loggingFolder="")
        esimateDepth = EsimateDepth()

        imageStartPath = "data/" + folder
        files = listdir(imageStartPath)
        files.sort()
        
        for fileName in tqdm(files):
            frame = read_image(imageStartPath + "/" +fileName, format="BGR") #bytt ut med PIL image read
            
            startTime = time.time()
            vis_panoptic, rgb_img, classImage = self.inferance.segmentImage(frame, fileName)
            diffTime = time.time() - startTime
            print("segmentering:", diffTime)

            startTime = time.time()
            #loggingFolder = "drivableImages/test"
            loggingFolder = ""
            drivableIndex, drivableColor = self.classToDrivable.imageClassesToDrivable(rgb_img, fileName, loggingFolder)
            diffTime = time.time() - startTime
            print("bildekovertering:", diffTime)

            if (useDepth):
                depthImage = esimateDepth.midasPredict(frame)
            else:
                depthImage=False

            startTime = time.time()
            self.segmented2Direction.getDirectionOfImage(drivableIndex, vis_panoptic, fileName, depthImage, rgb_img, drivableColor, useDepth=useDepth)
            diffTime = time.time() - startTime
            print("velg retning:", diffTime)
            #cv2.imwrite("test.png", depthImage)
            #break
            '''

    
    def runLidarBaselineFromFolder(self, folder, modelName, imageStartPath): #samle det som er likt i ein funksjon
        self.inferance = Inferance(loggingFolder="lidarSeg", modelName=modelName, imageStartPath=imageStartPath)
        self.classToDrivable = ClassesToDrivable()
        self.segmented2Direction = Segmented2Direction("lidar")

        imageStartPath = "data/" + folder
        files = listdir(imageStartPath)
        files.sort()
        self.inferance = Inferance(loggingFolder="lidarSeg", modelName=modelName, imageStartPath=imageStartPath)
        outputStartPath = "drivableImages/test"
        for fileName in tqdm(files):
            frame = read_image(imageStartPath + "/" +fileName, format="BGR") #bytt ut med PIL image read
            vis_panoptic, rgb_img, classImage = self.inferance.segmentImage(frame, fileName)
            drivableIndex, drivableColor = self.classToDrivable.imageClassesToDrivable(rgb_img, fileName, outputStartPath)
            self.segmented2Direction.getDirectionOfImage(drivableIndex, vis_panoptic, fileName, frame[:, :, 2], rgb_img, drivableColor, useDepth = True)
            #print(frame.shape)
            cv2.imwrite("segmentertLidar.png", rgb_img)
            #cv2.imwrite("originalLidar.png", rgb_img)

            #break
            


    def runRGBBaselineFromSubscribeBag(self):
        pass





#baseline = BaselineLive()#med bilde lagring, med datasetlagring
#baseline.runRGBBaselineFromFolder("inferanceInputImages", modelName = "semanticRGB", imageStartPath = "data/inferanceInputImages")
#baseline.runLidarBaselineFromFolder("inferanceLidar", modelName = "modelAltTilNo", imageStartPath = "data/inferanceLidar")



if __name__ == '__main__':

    typeOfBasline = sys.argv[1]
    goalX = int(sys.argv[2])
    goalY = int(sys.argv[3])
    print(typeOfBasline)
    
    rospy.init_node('baselineLive')

    if (typeOfBasline == "rgb"):
        node = BaselineLive("semanticRGB", useDepth=False, modelType=typeOfBasline, goalX=goalX, goalY=goalY) 
    elif (typeOfBasline == "lidar"):
        node = BaselineLive("semanticLiDAR", useDepth=True, modelType=typeOfBasline,  goalX=goalX, goalY=goalY)

    while not rospy.is_shutdown():
        rospy.spin()

    rospy.on_shutdown(node.on_shutdown)
