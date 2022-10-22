#!/usr/bin/env python
#
# ROS node to publish the true map from odom transform (map -> odom)
# using the ground truth location provided by gazebo and the
# known transform world from map (world -> map). The latter
# transform can be computed with print_world_map_coordinates.py
# and compute_world_map_transform.py.
#
# Author: Marynel Vazquez (marynelv@stanford.edu)
# Creation Date: 12/29/17

import copy

import rospy
import tf
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

# hacky matrix inverse because BLAS has an issue that makes it use all GPUs!!
# see bug here:
# https://github.com/numpy/numpy/issues/8120
#
# This inverse method is from here:
# https://stackoverflow.com/questions/32114054/matrix-inversion-without-numpy


def transposeMatrix(m):
    return map(list, zip(*m))


def getMatrixMinor(m, i, j):
    return [row[:j] + row[j + 1 :] for row in (m[:i] + m[i + 1 :])]


def getMatrixDeternminant(m):
    # base case for 2x2 matrix
    if len(m) == 2:

        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    determinant = 0
    for c in range(len(m)):
        determinant += ((-1) ** c) * m[0][c] * getMatrixDeternminant(getMatrixMinor(m, 0, c))
    return determinant


def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    # special case for 2x2 matrix:
    if len(m) == 2:
        return [
            [m[1][1] / determinant, -1 * m[0][1] / determinant],
            [-1 * m[1][0] / determinant, m[0][0] / determinant],
        ]

    # find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m, r, c)
            cofactorRow.append(((-1) ** (r + c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c] / determinant
    return cofactors


## end of hacky matrix inverse


class TrueMapOdomNode:
    """Node that publishes the true map -> odom transform"""

    def __init__(self):

        # vars
        self.ground_truth_odom = None

        # params
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.baselink_frame = rospy.get_param("~baselink_frame", "base_link")
        self.publish_rate = rospy.get_param("~publish_rate", 50)
        self.ground_truth_topic = rospy.get_param("~ground_truth_topic", "ground_truth_odom")
        self.publish_pose_in_map_marker = rospy.get_param("~pose_in_map_marker", "true")

        # connections
        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()

        if self.publish_pose_in_map_marker:
            self.marker_pub = rospy.Publisher("markers_true_pose", Marker, queue_size=10)

        # Publish the pose on a pseudo amcl ground truth topic
        self.pose_in_map_pub = rospy.Publisher("amcl_pose_ground_truth", PoseWithCovarianceStamped, queue_size=10)

        # get world->map static transform
        # (we do this only once since we know the transform is static)
        map_from_world_tup = None
        last_check = rospy.Time.now()
        while map_from_world_tup is None:
            rospy.sleep(0.1)

            try:
                self.tf_listener.waitForTransform(self.map_frame, self.world_frame, rospy.Time(0), rospy.Duration(0.1))
                map_from_world_tup = self.tf_listener.lookupTransform(
                    self.map_frame, self.world_frame, rospy.Time.now()
                )

                t_map_from_world, q_map_from_world = map_from_world_tup
                self.T_map_from_world = tf.transformations.translation_matrix(t_map_from_world)
                self.R_map_from_world = tf.transformations.quaternion_matrix(q_map_from_world)

            except (
                tf2_ros.TransformException,
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:

                if (rospy.Time.now() - last_check).to_sec() > 2.0:
                    rospy.logwarn(
                        "Waiting for {}->{} transform. Got exception {}.".format(self.world_frame, self.map_frame, e)
                    )

            last_check = rospy.Time.now()

        rospy.loginfo("Got static transform {}->{}".format(self.world_frame, self.map_frame))

        self.gt_sub = rospy.Subscriber(self.ground_truth_topic, Odometry, self.ground_truth_callback)

        rospy.spin()

    def got_frame(self, frame_name):
        """Query if a frame exists in the tf tree"""
        frame_exists = False
        try:
            frame_exists = self.tf_listener.frameExists(frame_name)
        except AttributeError:
            # This exception handling is necessary due to a bug in tf
            # see https://github.com/ros/geometry/issues/152
            pass
        return frame_exists

    def make_R_t_from_pose(self, pose):
        T = tf.transformations.translation_matrix([pose.position.x, pose.position.y, pose.position.z])
        Rq = tf.transformations.quaternion_matrix(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        )
        return Rq, T

    def ground_truth_callback(self, odometry_msg):

        # save groun truth odom
        self.ground_truth_odom = copy.deepcopy(odometry_msg)

        # convert ground truth pose (world->base_link) to the map frame
        # using the world->map static transform
        R_world_from_baselink, T_world_from_baselink = self.make_R_t_from_pose(self.ground_truth_odom.pose.pose)
        T_map_from_baselink = tf.transformations.concatenate_matrices(
            self.T_map_from_world, self.R_map_from_world, T_world_from_baselink, R_world_from_baselink
        )

        # separate the transform into the rotation and translation components
        t_baselink = tf.transformations.translation_from_matrix(T_map_from_baselink)
        q_baselink = tf.transformations.quaternion_from_matrix(T_map_from_baselink)

        # publish the baselink-in-map ground truth marker (should align with /amcl_pose visually)
        if self.publish_pose_in_map_marker:
            pose_in_map_marker = Marker()
            pose_in_map_marker.header.frame_id = self.map_frame
            pose_in_map_marker.header.stamp = rospy.Time.now()
            pose_in_map_marker.ns = "true_pose_markers"
            pose_in_map_marker.id = 0
            pose_in_map_marker.type = 0  # arrow
            pose_in_map_marker.pose.position.x = t_baselink[0]
            pose_in_map_marker.pose.position.y = t_baselink[1]
            pose_in_map_marker.pose.position.z = t_baselink[2]
            pose_in_map_marker.pose.orientation.x = q_baselink[0]
            pose_in_map_marker.pose.orientation.y = q_baselink[1]
            pose_in_map_marker.pose.orientation.z = q_baselink[2]
            pose_in_map_marker.pose.orientation.w = q_baselink[3]
            pose_in_map_marker.scale.x = 0.5  # arrow length
            pose_in_map_marker.scale.y = 0.1
            pose_in_map_marker.scale.z = 0.2
            pose_in_map_marker.color.r = 1.0
            pose_in_map_marker.color.g = 0.0
            pose_in_map_marker.color.b = 1.0
            pose_in_map_marker.color.a = 1.0
            pose_in_map_marker.lifetime = rospy.Time(0.5)
            self.marker_pub.publish(pose_in_map_marker)

        # publish baselink-in-map pose on /amcl_pose_ground_truth topic
        pose_in_map = PoseWithCovarianceStamped()
        pose_in_map.header.frame_id = self.map_frame
        pose_in_map.header.stamp = rospy.Time.now()
        pose_in_map.pose.pose.position.x = t_baselink[0]
        pose_in_map.pose.pose.position.y = t_baselink[1]
        pose_in_map.pose.pose.position.z = t_baselink[2]
        pose_in_map.pose.pose.orientation.x = q_baselink[0]
        pose_in_map.pose.pose.orientation.y = q_baselink[1]
        pose_in_map.pose.pose.orientation.z = q_baselink[2]
        pose_in_map.pose.pose.orientation.w = q_baselink[3]
        self.pose_in_map_pub.publish(pose_in_map)

        # now go in the other direction: compute map pose in base_link frame
        # T_baselink_from_map = tf.transformations.inverse_matrix(T_map_from_baselink)
        T_baselink_from_map = getMatrixInverse(T_map_from_baselink.tolist())

        # get the baselink -> odom transform
        try:
            self.tf_listener.waitForTransform(
                self.baselink_frame, self.odom_frame, odometry_msg.header.stamp, rospy.Duration(1.0)
            )

            t_baselink_from_odom, q_baselink_from_odom = self.tf_listener.lookupTransform(
                self.baselink_frame, self.odom_frame, odometry_msg.header.stamp
            )

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(e)
            return

        # compute the map from odom transform (map -> odom)
        T_map_from_odom = tf.transformations.concatenate_matrices(
            T_map_from_baselink,
            tf.transformations.translation_matrix(t_baselink_from_odom),
            tf.transformations.quaternion_matrix(q_baselink_from_odom),
        )

        t_map = tf.transformations.translation_from_matrix(T_map_from_odom)
        q_map = tf.transformations.quaternion_from_matrix(T_map_from_odom)

        rospy.logdebug("hello")
        self.tf_broadcaster.sendTransform(t_map, q_map, rospy.Time.now(), self.odom_frame, self.map_frame)


if __name__ == "__main__":
    rospy.init_node("publish_true_map_to_odom_tf", anonymous=True)

    try:
        mynode = TrueMapOdomNode()

    except rospy.ROSInitException:
        pass
