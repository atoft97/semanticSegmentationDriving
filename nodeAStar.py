from distutils.command.build import build
from re import search
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
import matplotlib.pyplot as plt

from heapq import heappush, heappop, heapify
#from PIL import Image

class NodeAStar:

    def __init__(self):
        print("initializing NodeAStar")
        
        #score += countDict[4] * (-10000) #obsticle 10000
        #score += countDict[3] * (-10) #verry bad terrain 1000
        #score += countDict[2] * (-1) #rough terrain 5
        #score += countDict[1] * (1) #good terrain 1
        #score += countDict[0] * (0) #unknown 10
        self.color_to_label = {(255, 255, 255): 10, (64, 255, 38): 5, (70, 70, 70): 10, (150, 0, 191): 10000, (255, 38, 38): 10000, (232, 227, 81): 1000, (255, 179, 0): 1000, (255, 20, 20):10000 , (191, 140, 0):1 , (15, 171, 255):10 , (200, 200, 200): 10000, (46, 153, 0): 10000, (180, 180, 180): 1}
        self.sub_lidarSections = rospy.Subscriber("/custom/mapImage", Image, self.findBestPath ,queue_size=1, buff_size=2*32)

        self.latest_xpos = None
        self.latest_ypos = None
        self.latest_zpos = None
        self.latets_heading = None
        self.bridge = CvBridge()

        self.counter = 0

        #self.pub_custom_pointcloud = rospy.Publisher("/custom/Pointcloud", PointCloud2, queue_size = 5)
        #self.pub_custom_rangeImage = rospy.Publisher("/custom/rangeImage", Image, queue_size = 5)
        #self.pub_custom_rangeReflect= rospy.Publisher("/custom/reflectImage", Image, queue_size = 5)
        #self.pub_custom_rangeSignal= rospy.Publisher("/custom/signalImage", Image, queue_size = 5)
        self.sub_pos_heading = rospy.Subscriber("/warpath/navigation/odometry_integrated_center_enu", Odometry, self.processHeadingAndPos,queue_size=1)

        self.pub_twist = rospy.Publisher("~cmd_vel", Twist, queue_size = 1)

        self.goalX = 0
        self.goalY = 0
        
        print("initialized NodeAStar")
    

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
        #self.latets_heading = (heading + 180) % 360
        self.latets_heading = (heading)




    def findBestPath(self, rosImage): #kan kjorast raskare enn kvart bilde for den for ny pos
        if (self.latest_xpos == None):
            return

        imageMapPosX = int(self.latest_xpos + 500)
        imageMapPosY = int(self.latest_ypos + 500)

        

        self.counter += 1
        image = np.array(self.bridge.imgmsg_to_cv2(rosImage))[...,::-1]
        cv2.imwrite("astar.png", image)

        #image_around_veichle = image[imageMapPosX-10:imageMapPosX+10, imageMapPosY-10:imageMapPosY+10]
        #image_around_veichle_large = np.zeros((1000,1000, 3))



        #cv2.imwrite("SmallAstar.png", image_around_veichle[...,::-1])
        #for i in range(-1,2):
        #    for j in range(-1,2):
        #        image[int(self.latest_xpos)+500+i, int(self.latest_ypos)+500+j] = (0,0,0)

        fig, ax = plt.subplots(nrows=1, figsize = (15,10))
        ax.scatter(self.goalY+500, self.goalX+500, zorder=1, c='b', s=10)
        ax.scatter(int(self.latest_ypos)+500, int(self.latest_xpos)+500, zorder=1, c='b', s=10)

        #ax[0].set_xlim(11.4040,11.4262)
        #ax[0].set_ylim(61.1740,61.1871)

        #print("heading", self.latets_heading)
        rad_actual = math.radians(self.latets_heading)
        x_heading = math.cos(rad_actual)*50  + imageMapPosX
        y_heading = math.sin(rad_actual)*50 + imageMapPosY

        #print(imageMapPosY, y_heading)
        #print(imageMapPosX, x_heading)

        
        ax.annotate("", xy=(imageMapPosY, imageMapPosX), xytext=(y_heading, x_heading), arrowprops=dict(arrowstyle="<-", color='blue'))

        #ax[0].annotate("", xy=(self.latest_xpo, latitude), xytext=(x_coord_actual, y_coord_actual), arrowprops=dict(arrowstyle="<-", color='red'))

        #ax[0].annotate("", xy=(self.latest_xpo, latitude), xytext=(x_coord_best, y_coord_best), arrowprops=dict(arrowstyle="<-", color='green'))

        #cv2.imwrite(f"astar/{self.counter}.png", image)
        #image[int(self.latest_xpos)+500, int(self.latest_ypos)+500] = (0,0,0)



        drivableMap = np.zeros((image.shape[0], image.shape[1]))
        for key in self.color_to_label.keys():
            drivableMap[(image == key).all(2)] = self.color_to_label[key]
        
        #print(drivableMap)
        #print(drivableMap[830:840, 570:580])

        cv2.imwrite("drivable.png", drivableMap)
        print("node a star recived new map", self.counter)
        #konverter til drivable
        
        #bestPath = self.astar(drivableMap, (int(self.latest_ypos)+500, int(self.latest_xpos)+500), (int(self.goalY)+500, int(self.goalX)+500)) #start in goal, switch
        #print("funnet best path", self.counter, bestPath)
        cv2.imwrite(f"astar2/{self.counter}withoutPath.png", image[...,::-1])
        #for node in bestPath:
        #    pos = node.pos
            #print(drivableMap[pos[1], pos[0]])
        #    image[pos[1], pos[0]] = (0,0,0)
        #print("plotted best path", self.counter, bestPath)
        #cv2.imwrite(f"astar2/{self.counter}withPath.png", image[...,::-1])
        ax.imshow(image, zorder=0, aspect= 'equal')
        indexOutputPath =  f'astar/{self.counter}'
        plt.savefig(indexOutputPath)
        plt.close()
        
    def heuristic_cost_estimate(self, a, b):
        length = abs(a[0] - b[0]) + abs(a[1] - b[1])
        length = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        return(length)

    def isGoalReached(self, a, b):
        return(a == b)

    def reconstruct_path(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current
                current = current.came_from
        if reversePath:
            return _gen()
        else:
            return reversed(list(_gen()))

    def get_neightbours(self, currentPos):
        neighboursPos = []
        for coorinatesChange in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: #mabye not diagonal (-1, -1), (-1, 1), (1, -1), (1, 1)
            node_position = (currentPos[0] + coorinatesChange[0], currentPos[1] + coorinatesChange[1])
            if (node_position[0] < 0 or node_position[0] >= 1000 or node_position[1] < 0 or node_position[1] >= 1000): 
                continue
            #remove obstacle early, bush 3 og tree 4
            neighboursPos.append(node_position)
        return(neighboursPos)

    def get_disconted_neigbour_cost(self, neighbourPos, costMap):
        #neigbourNeigbours = self.get_neightbours(neighbourPos)
        cost = 0
        for neigbourNeigbourPos in self.get_neightbours(neighbourPos):
            cost += costMap[neigbourNeigbourPos[1], neigbourNeigbourPos[0]] / 5
        return(cost)




    def astar(self, costMap, startPos, goalPos):
        #print(startPos, goalPos)

        if (self.isGoalReached(startPos, goalPos)):
            return([startPos])
        
        searchNodes = SearchNodeDict()
        startNode = searchNodes[startPos] = SearchNode(startPos, gscore=0, fscore=self.heuristic_cost_estimate(startPos, goalPos))
        openSet = []
        heappush(openSet, startNode)

        #print(searchNodes[startPos])
        #print(searchNodes[goalPos])

        while openSet:
            current = heappop(openSet)
            #print("current pos", current.pos)
            if (self.isGoalReached(current.pos, goalPos)):
                return(self.reconstruct_path(current))
            
            current.out_openset = True
            current.closed = True
            #neighbors = self.get_neightbours(current.pos)

            for neighbor in map(lambda n: searchNodes[n], self.get_neightbours(current.pos)):
                if (neighbor.closed):
                    continue
                #tentativeGscore = current.gscore + costMap[neighbor.pos[1], neighbor.pos[0]] + self.get_disconted_neigbour_cost(neighbor.pos, costMap)
                tentativeGscore = current.gscore + costMap[neighbor.pos[1], neighbor.pos[0]]
                #print(current.gscore, costMap[neighbor.pos[0], neighbor.pos[1]])
                #if (costMap[neighbor.pos[1], neighbor.pos[0]] != 10):
                #    print(neighbor.pos ,costMap[neighbor.pos[1], neighbor.pos[0]])
                if (tentativeGscore >= neighbor.gscore):
                    continue
                neighbor.came_from = current
                neighbor.gscore = tentativeGscore
                neighbor.fscore = tentativeGscore + self.heuristic_cost_estimate(neighbor.pos, goalPos)
                if (neighbor.out_openset):
                    neighbor.out_openset = False
                    heappush(openSet, neighbor)
                else:
                    openSet.remove(neighbor)
                    heappush(openSet, neighbor)


            #print(neighbors)
            #return ([])
        return (None)


    '''
        start_node = SearchNode(None, startPos)
        print("start node", start_node.position)
        start_node.g = 0
        start_node.h = 0
        start_node.f = 0

        end_node = SearchNode(None, goalPos)

        end_node.g = 0
        end_node.h = 0
        end_node.f = 0

        open_list = []
        closed_list = []

        open_list.append(start_node)

        while (len(open_list) > 0):
            current_node = open_list[0]
    '''


        
Infinite = float('inf')

class SearchNode:
        __slots__ = ('pos', 'gscore', 'fscore',
                     'closed', 'came_from', 'out_openset')

        def __init__(self, pos, gscore=float('inf'), fscore=float('inf')):
            self.pos = pos
            self.gscore = gscore
            self.fscore = fscore
            self.closed = False
            self.out_openset = True
            self.came_from = None

        def __lt__(self, b):
            return self.fscore < b.fscore

class SearchNodeDict(dict):

    def __missing__(self, k):
        v = SearchNode(k)
        self.__setitem__(k, v)
        return v


if __name__ == '__main__':
    rospy.init_node('nodeAStar')

    node = NodeAStar() 

    while not rospy.is_shutdown():
        rospy.spin()

    rospy.on_shutdown(node.on_shutdown)
