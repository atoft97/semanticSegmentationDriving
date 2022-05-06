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

        
        #self.esimateDepth = EsimateDepth()

        #self.latest_longitude = 0
        #self.latest_latitude = 0 
        #self.latets_heading = 0
        self.color_to_label = {(255, 255, 255): 0, (64, 255, 38): 1, (70, 70, 70): 2, (150, 0, 191): 3, (255, 38, 38): 4, (232, 227, 81): 5, (255, 179, 0): 6, (255, 20, 20):7 , (191, 140, 0):8 , (15, 171, 255):9 , (200, 200, 200): 10, (46, 153, 0): 11, (180, 180, 180): 12}
        #self.label_to_color = {0: [255, 255, 255], 1: [64, 255, 38], 2: [70, 70, 70], 3: [150, 0, 191], 4: [255, 38, 38], 5: [232, 227, 81], 6: [255, 179, 0], 7: [255, 20, 20], 8: [191, 140, 0], 9: [15, 171, 255], 10: [200, 200, 200], 11: [46, 153, 0], 12: [180, 180, 180]}
        self.label_to_color_nonMovable = {0: [255, 255, 255], 1: [64, 255, 38], 2: [70, 70, 70], 3: [150, 0, 191],  5: [232, 227, 81], 6: [255, 179, 0], 7: [255, 20, 20], 8: [191, 140, 0], 10: [200, 200, 200], 11: [46, 153, 0], 12: [180, 180, 180]}
        self.label_to_color_movable = {4: [255, 38, 38]}

        
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

        self.combinedMostLikely = np.zeros((1000, 1000))

        self.numpyMap = np.zeros((15, 1000, 1000))
        self.mostLikelyMap = np.zeros((1000, 1000, 3))
        self.mostLikelyMap[:,:] = (255,255,255)
        print(self.numpyMap)
        print(self.numpyMap.shape)
        

        #self.sub_cloud = rospy.Subscriber("/ugv_sensors/lidar/cloud/points", PointCloud2, self.processLidarImage,queue_size=1) #konverter sjølv med ouster eller ta inn 4 topics (/ugv_sensors/lidar/image/range_image)
        #255
        stortTall = 2**32
        #print("stor", stortTall)

        if (modelType=="rgb"):
            self.sub_rgb = rospy.Subscriber("/ugv_sensors/camera/color/image", Image, self.processRGBImage,queue_size=1, buff_size=stortTall)
        elif (modelType=="lidar"):
            #self.sub_range = rospy.Subscriber("/ugv_sensors/lidar/image/range_image", Image, self.processRange,queue_size=1, buff_size=stortTall)
            #self.sub_signal = rospy.Subscriber("/ugv_sensors/lidar/image/signal_image", Image, self.processSignal,queue_size=1, buff_size=stortTall)
            #self.sub_reflec = rospy.Subscriber("/ugv_sensors/lidar/image/reflec_image", Image, self.processReflec,queue_size=1, buff_size=stortTall)
            #self.sub_lidar = rospy.Subscriber("/ugv_sensors/lidar/cloud/points", PointCloud2, self.processLidar,queue_size=1, buff_size=stortTall)
            self.sub_range = rospy.Subscriber("/custom/rangeImage", Image, self.processRange,queue_size=1, buff_size=stortTall)
            self.sub_signal = rospy.Subscriber("/custom/signalImage", Image, self.processSignal,queue_size=1, buff_size=stortTall)
            self.sub_reflec = rospy.Subscriber("/custom/reflectImage", Image, self.processReflec,queue_size=1, buff_size=stortTall)
            self.sub_lidar = rospy.Subscriber("/custom/Pointcloud", PointCloud2, self.processLidar,queue_size=1, buff_size=stortTall)
            
            #self.sub_lidar_imu = rospy.Subscriber("/ugv_sensors/lidar/cloud/imu", Imu, self.processLidarImu,queue_size=1, buff_size=stortTall)
            

        #self.sub_lidar = rospy.Subscriber("/ugv_sensors/lidar/cloud/points", PointCloud2, self.processLidar,queue_size=1, buff_size=stortTall)
        #self.sub_pos = rospy.Subscriber("/ugv_sensors/navp_ros/nav_sat_fix", NavSatFix, self.processPOS,queue_size=1)
        #self.sub_pos = rospy.Subscriber("/warpath/navigation/odometry_integrated_center_enu", NavSatFix, self.processPOS,queue_size=1)
        #self.sub_heading = rospy.Subscriber("/warpath/navigation/odometry_integrated_center_enu", Odometry, self.processHeading,queue_size=1)
        self.sub_pos_heading = rospy.Subscriber("/warpath/navigation/odometry_integrated_center_enu", Odometry, self.processHeadingAndPos,queue_size=1)
        #self.sub_heading = rospy.Subscriber("/ugv_sensors/navp_ros/navp_msg", Odometry, self.processHeading,queue_size=1)
        self.bridge = CvBridge()

        self.pub_twist = rospy.Publisher("~cmd_vel", Twist, queue_size = 1)
        self.pub_segmented_image = rospy.Publisher("~segmentedImage", Image, queue_size = 1)
        self.pub_segmented_pointcloud = rospy.Publisher("~segmentedPointCloud", PointCloud2, queue_size = 1)
        self.pub_map_image = rospy.Publisher("/custom/mapImage", Image, queue_size = 1)

        #self.freq = rospy.get_param("~freq", 5.)
        #print("freq", self.freq)
        if (modelType=="rgb"):
            self.timer = rospy.Timer(rospy.Duration(1./10), self.timerFunction)
        elif (modelType=="lidar"):
            #self.timerLidar = rospy.Timer(rospy.Duration(1./10), self.timerLidar)
            self.timeLidarPointCloud = rospy.Timer(rospy.Duration(1./10), self.timeLidarPointCloud)
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

    def processLidar(self, rosPointcloud):
        #print("lidar:", rosPointcloud.header.seq)
        self.latest_pointCloud = rosPointcloud
        self.lindarSyncedOdom = self.latest_rosOdom

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
        if (rosImage == None):
            #print("tomt bilde")
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
            image = self.bridge.imgmsg_to_cv2(rosImage, desired_encoding="bgr8")
            
            #print(type(image))

            vis_panoptic, rgb_img, classImage = self.inferance.segmentImage(image, fileName)
            cv2.imwrite("testRGB.png", rgb_img)

            if (useDepth):
                depthImage = self.esimateDepth.midasPredict(image) #add stero vision
            else:
                depthImage=False

            #cv2.imwrite("outputData/rgbImages/test" +str(self.imageCounter)+".png", rgb_img)
            drivableIndex, drivableColor = self.classToDrivable.imageClassesToDrivableimage(rgb_img, fileName, loggingFolder)


            recomendedDirection = self.segmented2Direction.getDirectionOfImage(drivableIndex, vis_panoptic, fileName, depthImage, rgb_img, drivableColor, useDepth=useDepth, lat=self.latest_xpos, long=self.latest_ypos, heading=self.latets_heading)
            print(recomendedDirection)

            #send twist            
            twist = Twist()
            twist.linear.x = 1 #constant low speed for testing
            twist.angular.z = math.radians(recomendedDirection) # mabye change to negative for counter clockwise
            self.pub_twist.publish(twist)

            rgb_img = rgb_img.astype('uint8')
            self.pub_segmented_image.publish(self.bridge.cv2_to_imgmsg(rgb_img))


            

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

        odom = copy.copy(self.lindarSyncedOdom)

        if (rosRange == None or rosReflec==None or rosSignal==None or rosPointCLoud==None or odom==None):
            #print("tomt bilde")
            return

        rosY = odom.pose.pose.position.y
        rosX = odom.pose.pose.position.x
        rosZ = odom.pose.pose.position.z

        rosRotationX = odom.pose.pose.orientation.x
        rosRotationY = odom.pose.pose.orientation.y
        rosRotationZ = odom.pose.pose.orientation.z
        rosRotationW = odom.pose.pose.orientation.w


        print(rosReflec.header)
        print(rosRange.header)
        print(rosSignal.header)
        print(rosPointCLoud.header)
        print(odom.header)

        if (self.ready):

            self.ready = False

            self.imageCounter +=1 
            lokalCounter = self.imageCounter
            print("start", lokalCounter)

            fileName = "tmp.png"
            loggingFolder = ""
            useDepth = False

            #print(rosRange)
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
            
            combined = cv2.merge((signalImageBright, reflecImageBright, rangeImageBright))
            cv2.imwrite("testCombined.png", combined)

            cv2.imwrite("testRange.png", rangeImageBright)
            cv2.imwrite("testRefelc.png", reflecImageBright)
            cv2.imwrite("testSignal.png", signalImageBright)

            width = combined.shape[1] # keep original width
            height = 64*4
            dim = (width, height)
            combinedTaller = cv2.resize(combined, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite("testCombined.png", combinedTaller)
            
            vis_panoptic, rgb_img, classImage = self.inferance.segmentImage(combinedTaller, fileName)
            cv2.imwrite("testLidar.png", rgb_img)

            #if (useDepth):
            #    depthImage = self.esimateDepth.midasPredict(image) #add stero vision
            #else:
            #    depthImage=False

            #cv2.imwrite("outputData/rgbImages/test" +str(self.imageCounter)+".png", rgb_img)
            drivableIndex, drivableColor = self.classToDrivable.imageClassesToDrivable(rgb_img, fileName, loggingFolder)

            #(drivableIndex, vis_panoptic, fileName, frame[:, :, 2], rgb_img, drivableColor, useDepth = True)
            recomendedDirection = self.segmented2Direction.getDirectionOfImage(drivableIndex, vis_panoptic, fileName, combinedTaller[:,:,2], rgb_img, drivableColor, useDepth=True, lat=self.latest_xpos, long=self.latest_ypos, heading=self.latets_heading)
            print(recomendedDirection)


            #rgb bilde mindre
            width = rgb_img.shape[1] # keep original width
            height = 64
            dim = (width, height)

            rgbSmaller = cv2.resize(rgb_img, dim, interpolation = cv2.INTER_NEAREST)
            rgbSmaller = cv2.cvtColor(rgbSmaller.astype('float32'), cv2.COLOR_BGR2RGB)
            segmentation_img_staggered = client.destagger(self.metadataLidar, rgbSmaller, inverse=True)
            cv2.imwrite("stagg.png", segmentation_img_staggered)
            segmentation_img_staggeredFloat = (segmentation_img_staggered.astype(float) / 255.0).reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            numpyPointcloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(rosPointCLoud, remove_nans=True)
            

            '''
            #numpyPointcloud[:, 2] = 0
            print(numpyPointcloud)

            pcd.points = o3d.utility.Vector3dVector(numpyPointcloud)
            pcd.colors = o3d.utility.Vector3dVector(segmentation_img_staggeredFloat)

            print("x and y an z", rosX, rosY, rosZ)
            eulerOdom = euler_from_quaternion((rosRotationX, rosRotationY, rosRotationZ, rosRotationW))
            xrotation = math.degrees(eulerOdom[0])
            yrotation = math.degrees(eulerOdom[1])
            zrotation = math.degrees(eulerOdom[2])

            eulerDegrees = [xrotation, yrotation, zrotation]
            quaternion = quaternion_from_euler(math.radians(xrotation), math.radians(yrotation), math.radians(zrotation))

            print(eulerOdom)
            print("kjoretoys rotasjon:", eulerOdom)
            print("kjoretoy degrees:", eulerDegrees)
            xrotationEtter = 180-math.degrees(eulerOdom[2])
            yrotationEtter = 0
            zrotationEtter = 0

            eulerDegreesEtter = [xrotationEtter, yrotationEtter, zrotationEtter]
            print("Function: Rotation etter degrees: ", eulerDegreesEtter)
            quaternion = quaternion_from_euler(math.radians(xrotationEtter), math.radians(yrotationEtter), math.radians(zrotationEtter))

            R_odom = pcd.get_rotation_matrix_from_quaternion(quaternion)
            pcd.rotate(R_odom, center=(0,0,0))
            pcd.translate([rosX, rosY, rosZ])

            #pcd = pcd.voxel_down_sample(voxel_size=0.1)
            pcd = pcd.remove_non_finite_points()
            #self.combinedPointCLoud += pcd

            rosPontCloud = self.o3dpc_to_rospc(pcd, rosPointCLoud.header)
            self.pub_segmented_pointcloud.publish(rosPontCloud)
            '''


            numpyPointcloud[:, 2] = 0
            print(numpyPointcloud)

            pcd.points = o3d.utility.Vector3dVector(numpyPointcloud)
            pcd.colors = o3d.utility.Vector3dVector(segmentation_img_staggeredFloat)

            print("x and y an z", rosX, rosY, rosZ)
            eulerOdom = euler_from_quaternion((rosRotationX, rosRotationY, rosRotationZ, rosRotationW))
            xrotation = math.degrees(eulerOdom[0])
            yrotation = math.degrees(eulerOdom[1])
            zrotation = math.degrees(eulerOdom[2])

            eulerDegrees = [xrotation, yrotation, zrotation]
            quaternion = quaternion_from_euler(math.radians(xrotation), math.radians(yrotation), math.radians(zrotation))

            print(eulerOdom)
            print("kjoretoys rotasjon:", eulerOdom)
            print("kjoretoy degrees:", eulerDegrees)
            xrotationEtter = 180-math.degrees(eulerOdom[2])
            yrotationEtter = 0
            zrotationEtter = 0

            eulerDegreesEtter = [xrotationEtter, yrotationEtter, zrotationEtter]
            print("Function: Rotation etter degrees: ", eulerDegreesEtter)
            quaternion = quaternion_from_euler(math.radians(xrotationEtter), math.radians(yrotationEtter), math.radians(zrotationEtter))

            R_odom = pcd.get_rotation_matrix_from_quaternion(quaternion)
            pcd.rotate(R_odom, center=(0,0,0))
            pcd.translate([rosX, rosY, rosZ])

            #pcd = pcd.voxel_down_sample(voxel_size=0.1)
            pcd = pcd.remove_non_finite_points()
            #self.combinedPointCLoud += pcd
            rosPontCloud = self.o3dpc_to_rospc(pcd, rosPointCLoud.header)
            self.pub_segmented_pointcloud.publish(rosPontCloud)

            
            #self.combinedPointCLoud = self.combinedPointCLoud.voxel_down_sample(voxel_size=0.1)

            
            
            #if (self.imageCounter == 10):
            #    o3d.visualization.draw_geometries([pcd])
                #voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.combinedPointCLoud,
                #                                                voxel_size=0.5)
                #o3d.visualization.draw([voxel_grid])


            pointsWithCoordiantes = np.asarray(pcd.points)
            pointsWithColors = np.asarray(pcd.colors)
            #print(pcd.points)
            
            #print(pointsWithCoordiantes)
            
            print(pointsWithCoordiantes.shape)
            print("punkt start")

            numpyMapMovable = np.zeros((15, 1000, 1000))
            for i in range(pointsWithCoordiantes.shape[0]):
                try:
                    coorinates = pointsWithCoordiantes[i]
                    color = pointsWithColors[i]
                    #print(color)
                    color = tuple(np.floor(color*255).astype(int))
                    #print(color)
                    label = self.color_to_label[color] 
                    if (label == 4): # not in immovable
                        numpyMapMovable[label, int(coorinates[0])+500, int(coorinates[1])+500] += 1
                    elif(label != 9): #not sky, points are never sky, only missclassifications
                        self.numpyMap[label, int(coorinates[0])+500, int(coorinates[1])+500] += 1
                except:
                    print("unknown color")
                    
            #self.numpyMap = self.numpyMap/5
            print("punkt slutt")
            #self.numpyMap
            #print(self.numpyMap)

            mostLikelyClassNonMovable = self.numpyMap.argmax(axis=0)
            mostLikelyClassMovable = numpyMapMovable.argmax(axis=0)
            #print("most likely", mostLikelyClass.shape)

            #print(mostLikelyClass)

            #self.mostLikelyMap[mostLikelyClass != 0] =  4


            #rgb_img_most_likely_map = np.zeros((*self.mostLikelyMap.shape, 3))
            #self.mostLikelyMap
            '''
            for key in self.label_to_color.keys():
                self.mostLikelyMap[mostLikelyClass == key] = self.label_to_color[key]
            self.mostLikelyMap = cv2.cvtColor(self.mostLikelyMap.astype('float32'), cv2.COLOR_BGR2RGB)
            #print(rgb_img_most_likely)
            cv2.imwrite("mostLikelyMap.png", self.mostLikelyMap)
            '''

            #if visualize
            rgb_img_most_likely = np.zeros((*mostLikelyClassNonMovable.shape, 3))
            #rgb_img_most_likely[:,:] = (255,255,255)
            for key in self.label_to_color_nonMovable.keys():
                rgb_img_most_likely[mostLikelyClassNonMovable == key] = self.label_to_color_nonMovable[key]

            for key in self.label_to_color_movable.keys():
                rgb_img_most_likely[mostLikelyClassMovable == key] = self.label_to_color_movable[key]

            rgb_img_most_likely = cv2.cvtColor(rgb_img_most_likely.astype('float32'), cv2.COLOR_BGR2RGB)
            print(rgb_img_most_likely)
            cv2.imwrite("mostLikely.png", rgb_img_most_likely)
            rgb_img_most_likely = rgb_img_most_likely.astype('uint8')
            self.pub_map_image.publish(self.bridge.cv2_to_imgmsg(rgb_img_most_likely))

            #send twist            
            twist = Twist()
            twist.linear.x = 1 #constant low speed for testing
            twist.angular.z = -math.radians(recomendedDirection) # mabye change to negative for counter clockwise
            self.pub_twist.publish(twist)

            #outputImage = Image()
            rgb_img = rgb_img.astype('uint8')
            self.pub_segmented_image.publish(self.bridge.cv2_to_imgmsg(rgb_img))
            

            #o3d.visualization.draw_geometries([pcd])

            #outputImage = Image()
            #self.pub_segmented_image.publish(rgb_img)

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
        
    def processRGBImage(self, rosImage):
        self.latest_image = rosImage


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
