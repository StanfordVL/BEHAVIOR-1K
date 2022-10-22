#!/usr/bin/python
import numpy as np
import rospy
from geometry_msgs.msg import Twist


class NoiseInjectionNode:
    def __init__(self):
        rospy.init_node("gibson-sim-noise")
        self.register_callback()
        self.pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=10)

        self.x, self.z = 0, 0

    def register_callback(self):
        rospy.Subscriber("/mobile_base/commands/velocity_raw", Twist, self.callback)

    def callback(self, msg):
        print(msg)

        if not (msg.linear.x == 0 and msg.angular.z == 0):
            msg.linear.x += self.x
            msg.angular.z += self.z

            self.x = self.x * 0.95 + np.random.normal(0, 0.2) * 0.05
            self.z = self.z * 0.95 + np.random.normal(0, 1) * 0.05

        # pass through for zero velocity command

        self.pub.publish(msg)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = NoiseInjectionNode()
    node.run()
