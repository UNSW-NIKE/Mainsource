#! /usr/bin/python
"a simple rospy that can publisher stop as if the robot is currently nearby the start position"
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import datetime


def call_back(msg):
    #rospy.loginfo(msg.pose.pose.position)
    pos_x = msg.pose.pose.position.x
    pos_y = msg.pose.pose.position.y
    pos_z = msg.pose.pose.position.z
    ori_x = msg.pose.pose.orientation.x
    ori_y = msg.pose.pose.orientation.y
    ori_z = msg.pose.pose.orientation.z
    cmd_pub = rospy.Publisher('/cmd', String, queue_size=10)
    value = (pos_x - ori_x)**2 + (pos_y - ori_y)**2 + (pos_z - ori_z)**2
    rospy.loginfo(value)
    if value <= 0.1: #p1
        cmd_pub.publish("stop")


def send_stop():
    #timer
    start_time = datetime.datetime.now()
    rospy.init_node('send_stop_node')

    #WAIT AFTER 10 SECS
    while True:
        end_time = datetime.datetime.now()
        interval = (end_time - start_time).seconds
        rospy.loginfo(interval)#test
        if interval > 10: #p2
            break

    rospy.Subscriber("odom",Odometry,call_back)
    rospy.spin()


if __name__ == "__main__":
    send_stop()


