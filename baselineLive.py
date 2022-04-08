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
from sensor_msgs.msg import PointCloud2, Image, NavSatFix
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from cv_bridge import CvBridge
import numpy as np

from tf.transformations import euler_from_quaternion
import math

#from ouster import client
import sys

#from PIL import Image

class BaselineLive:

    def __init__(self, modelName, useDepth, modelType,  goalX, goalY):
        print("initializing")
        self.inferance = Inferance(loggingFolder="", modelName=modelName)
        self.classToDrivable = ClassesToDrivable()
        #self.segmented2Direction = Segmented2Direction("rgb", csvPath="outputDataset/rgbDriving.csv", loggingFolder="")
        #self.segmented2Direction = Segmented2DirectionLocal("rgb", csvPath="outputDataset/rgbDriving.csv", loggingFolder="")
        self.segmented2Direction = Segmented2DirectionLocal(modelType, csvPath="outputDataset/rgbDriving.csv", loggingFolder="", goalX=goalX, goalY=goalY)

        
        #self.esimateDepth = EsimateDepth()

        #self.latest_longitude = 0
        #self.latest_latitude = 0 
        #self.latets_heading = 0

        self.latest_xpos = 0
        self.latest_ypos = 0 
        self.latets_heading = 0
        self.latest_image = None
        self.last_processed_image = None
        self.latest_reflec = None
        self.latest_range = None
        self.latest_signal = None

        #self.sub_cloud = rospy.Subscriber("/ugv_sensors/lidar/cloud/points", PointCloud2, self.processLidarImage,queue_size=1) #konverter sjølv med ouster eller ta inn 4 topics (/ugv_sensors/lidar/image/range_image)
        
        stortTall = 2**32
        #print("stor", stortTall)

        if (modelType=="rgb"):
            self.sub_rgb = rospy.Subscriber("/ugv_sensors/camera/color/image", Image, self.processRGBImage,queue_size=1, buff_size=stortTall)
        elif (modelType=="lidar"):
            self.sub_range = rospy.Subscriber("/ugv_sensors/lidar/image/range_image", Image, self.processRange,queue_size=1, buff_size=stortTall)
            self.sub_signal = rospy.Subscriber("/ugv_sensors/lidar/image/signal_image", Image, self.processSignal,queue_size=1, buff_size=stortTall)
            self.sub_reflec = rospy.Subscriber("/ugv_sensors/lidar/image/reflec_image", Image, self.processReflec,queue_size=1, buff_size=stortTall)

        #self.sub_lidar = rospy.Subscriber("/ugv_sensors/lidar/cloud/points", PointCloud2, self.processLidar,queue_size=1, buff_size=stortTall)
        #self.sub_pos = rospy.Subscriber("/ugv_sensors/navp_ros/nav_sat_fix", NavSatFix, self.processPOS,queue_size=1)
        #self.sub_pos = rospy.Subscriber("/warpath/navigation/odometry_integrated_center_enu", NavSatFix, self.processPOS,queue_size=1)
        #self.sub_heading = rospy.Subscriber("/warpath/navigation/odometry_integrated_center_enu", Odometry, self.processHeading,queue_size=1)
        self.sub_pos_heading = rospy.Subscriber("/warpath/navigation/odometry_integrated_center_enu", Odometry, self.processHeadingAndPos,queue_size=1)
        #self.sub_heading = rospy.Subscriber("/ugv_sensors/navp_ros/navp_msg", Odometry, self.processHeading,queue_size=1)
        self.bridge = CvBridge()

        self.pub_twist = rospy.Publisher("~cmd_vel", Twist, queue_size = 1)
        self.pub_segmented_image = rospy.Publisher("~segmentedImage", Image, queue_size = 1)

        #self.freq = rospy.get_param("~freq", 5.)
        #print("freq", self.freq)
        if (modelType=="rgb"):
            self.timer = rospy.Timer(rospy.Duration(1./10), self.timerFunction)
        elif (modelType=="lidar"):
            self.timerLidar = rospy.Timer(rospy.Duration(1./10), self.timerLidar)
        #self.timer = rospy.Timer(rospy.Duration(1./10), self.timerFunction)
        
        #print(self.timer)

        #metadata_path = "lidar_metadata.json"

        #with open(metadata_path, 'r') as f:
        #    metadata = client.SensorInfo(f.read())

        print("initialized")
        self.imageCounter = 0

        self.ready = True

        self.imageCounter = 0

    #def processLidar(self, rosPointcloud):

    def processRange(self, rosImage):
        #print("new range")
        self.latest_range = rosImage
    
    def processSignal(self, rosImage):
        self.latest_signal = rosImage
    
    def processReflec(self, rosImage):
        self.latest_reflec = rosImage

    
    def processHeadingAndPos(self, rosOdom):
        self.latest_xpos = rosOdom.pose.pose.position.x
        self.latest_ypos = rosOdom.pose.pose.position.y
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
            drivableIndex, drivableColor = self.classToDrivable.imageClassesToDrivable(rgb_img, fileName, loggingFolder)


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

    
    def timerLidar(self, event=None):
        #print("timer")
        #rosImage = self.latest_image

        rosReflec = self.latest_reflec
        rosRange = self.latest_range
        rosSignal= self.latest_signal 

        if (rosRange == None or rosReflec==None or rosSignal==None):
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
            recomendedDirection = self.segmented2Direction.getDirectionOfImage(drivableIndex, vis_panoptic, fileName, combinedTaller[:,:,2], rgb_img, drivableColor, useDepth=True, lat=self.latest_xpos, long=self.latest_ypos, heading=self.latets_heading)
            print(recomendedDirection)

            #send twist            
            twist = Twist()
            twist.linear.x = 1 #constant low speed for testing
            twist.angular.z = math.radians(recomendedDirection) # mabye change to negative for counter clockwise
            self.pub_twist.publish(twist)

            #outputImage = Image()
            rgb_img = rgb_img.astype('uint8')
            self.pub_segmented_image.publish(self.bridge.cv2_to_imgmsg(rgb_img))
            

            #outputImage = Image()
            #self.pub_segmented_image.publish(rgb_img)

            #self.ready = True

            print("end", lokalCounter)
            print("\n")
            self.ready = True
            

        
    def processRGBImage(self, rosImage):

        self.latest_image = rosImage
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
