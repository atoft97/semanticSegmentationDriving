from LiDARsegmentation.maskFormer.inferance.inferance import Inferance
from tqdm import tqdm
from os import listdir
from detectron2.data.detection_utils import read_image
from driving.classesToDrivable import ClassesToDrivable
import cv2
from driving.segmented2DIrection import Segmented2Direction
import time
#from monoDepth.esimateDepth import EsimateDepth
import rospy
from sensor_msgs.msg import PointCloud2, Image, NavSatFix
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge

class BaselineLive:

    def __init__(self, modelName, useDepth=False):
        print("initializing")
        self.inferance = Inferance(loggingFolder="", modelName=modelName)
        self.classToDrivable = ClassesToDrivable()
        self.segmented2Direction = Segmented2Direction("rgb", csvPath="outputDataset/rgbDriving.csv", loggingFolder="")
        #self.esimateDepth = EsimateDepth()

        self.latest_longitude = 0
        self.latest_latitude = 0 
        self.latets_heading = 0

        #self.sub_cloud = rospy.Subscriber("/ugv_sensors/lidar/cloud/points", PointCloud2, self.processLidarImage,queue_size=1) #konverter sjølv med ouster eller ta inn 4 topics (/ugv_sensors/lidar/image/range_image)
        self.sub_rgb = rospy.Subscriber("/ugv_sensors/camera/color/image", Image, self.processRGBImage,queue_size=1)
        self.sub_rgb = rospy.Subscriber("/ugv_sensors/navp_ros/nav_sat_fix", NavSatFix, self.processPOS,queue_size=1)
        self.sub_rgb = rospy.Subscriber("/ugv_sensors/navp_ros/navp_msg", rospy.AnyMsg, self.processHeading,queue_size=1)
        self.bridge = CvBridge()

        self.pub_twist = rospy.Publisher("~speedAndDirection", Twist, queue_size = 1)
        self.pub_segmented_image = rospy.Publisher("~segmentedImage", Image, queue_size = 1)

        print("initialized")
        self.imageCounter = 0
    
    def processPOS(self, rosPos):
        #print(rosPos)
        self.latest_longitude = rosPos.longitude
        self.latest_latitude = rosPos.latitude
        
        
    def processHeading(self, data):
        #print(type(data))
        #print(data)
        #print(dir(data))
        connection_header = data._connection_header['type'].split('/')
        ros_pkg = connection_header[0] + '.msg'
        msg_type = connection_header[1]
        #print(msg_type)
        #print(ros_pkg)
        #print(data._full_text)
        heading = 200 #temporary fix before i can read from custum topic
        self.latets_heading = heading
        #msg_class = getattr(import_module(ros_pkg), msg_type)
        
    def processRGBImage(self, rosImage):
        fileName = "tmp.png"
        loggingFolder = ""
        useDepth = False
        #if ferdigprossesert, gå videre
        #print("bidle motatt")
        #print(type(rosImage))
        image = self.bridge.imgmsg_to_cv2(rosImage, desired_encoding="bgr8")
        #print(type(image))

        vis_panoptic, rgb_img, classImage = self.inferance.segmentImage(image, fileName)

        if (useDepth):
            depthImage = self.esimateDepth.midasPredict(image) #add stero vision
        else:
            depthImage=False

        #cv2.imwrite("outputData/rgbImages/test" +str(self.imageCounter)+".png", rgb_img)
        drivableIndex, drivableColor = self.classToDrivable.imageClassesToDrivable(rgb_img, fileName, loggingFolder)


        recomendedDirection = self.segmented2Direction.getDirectionOfImage(drivableIndex, vis_panoptic, fileName, depthImage, rgb_img, drivableColor, useDepth=useDepth, lat=self.latest_latitude, long=self.latest_longitude, heading=self.latets_heading)
        print(recomendedDirection)

        #send twist            
        twist = Twist()
        twist.linear = 5 #constant low speed for testing
        twist.angular = recomendedDirection

        self.pub_twist.publish(twist)

        #outputImage = Image()
        #self.pub_segmented_image.publish(rgb_img)
        self.imageCounter += 1







#baseline = BaselineLive()#med bilde lagring, med datasetlagring
#baseline.runRGBBaselineFromFolder("inferanceInputImages", modelName = "semanticRGB", imageStartPath = "data/inferanceInputImages")
#baseline.runLidarBaselineFromFolder("inferanceLidar", modelName = "modelAltTilNo", imageStartPath = "data/inferanceLidar")



if __name__ == '__main__':
    
    rospy.init_node('baselineLive')

    node = BaselineLive("semanticRGB", useDepth=False)

    while not rospy.is_shutdown():
        rospy.spin()

    rospy.on_shutdown(node.on_shutdown)
