import numpy as np
import rospy
from std_msgs.msg import Float32, Int64
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped, Point
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from nav_msgs.msg import Odometry, Path
import message_filters
import rospkg
from cv_bridge import CvBridge
import cv2
import struct
from tf import TransformListener, transformations
import time

class Challenge(object):
    def __init__(self):
        self.image_width = 160
        self.image_height = 90
        #self.image_width_raw = 640
        #self.image_height_raw = 480
        self.depth_min = 0.1
        self.depth_max = 10.0
        self.sensor_dim = 4

        self.obs = {
            'depth': np.ones((self.image_height, self.image_width, 1)),
            'rgb': np.ones((self.image_height, self.image_width, 3)),
            'sensor': np.ones((self.sensor_dim))
        }
        self.pose = None
        rospy.init_node('challenge-real') #initialize ros node
        self.tf_listener_ = TransformListener()
        print('tf_listener')

        self.velocity_publisher = rospy.Publisher('/cmd_vel_mux/input/navigation_nn', Twist, queue_size=10)
        print('velocity_publisher')

        self.cv_bridge = CvBridge()
        depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
        rgb_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub],
                                                                queue_size=1,
                                                                slop=0.5)
        self.sync.registerCallback(self.sync_callback)
        self.sensor_ready = False 

        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        self.goal = None
        self.goal_ready = False
        #self.goal = np.array([3.0, 3.0, 0.0])
        #self.goal_ready = True

    def fromTranslationRotation(self, translation, rotation):
        """
        :param translation: translation expressed as a tuple (x,y,z)
        :param rotation: rotation quaternion expressed as a tuple (x,y,z,w)
        :return: a :class:`numpy.matrix` 4x4 representation of the transform
        :raises: any of the exceptions that :meth:`~tf.Transformer.lookupTransform` can raise

        Converts a transformation from :class:`tf.Transformer` into a representation as a 4x4 matrix.
        """
        return np.dot(transformations.translation_matrix(translation), transformations.quaternion_matrix(rotation))

    def goal_callback(self, msg):
        position = msg.pose.position
        self.goal = np.array([position.x, position.y, position.z])
        self.goal_ready = True

    def amcl_callback(self, msg):
        #print('amcl_callback')
        self.pose = msg.pose.pose
        print(self.pose)
        #print(time.time() - self.last_time_pose_update)
        #print(self.pose)
        # self.last_time_pose_update = time.time()

        #t = self.tf_listener_.getLatestCommonTime("/base_link", "/map")
        #(trans, rot) = listener.lookupTransform('/map', 'base_link', t)
        #rot = transformations.quaternion_matrix([self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w])


        #(lin, ang) = self.tf_listener_.lookupTwistFull('/base_link', '/map', '/base_link', (0,0,0), '/map', rospy.Time(0.0), rospy.Duration(0.5))
       
            
            #yaw = -rot[2]
            #lin_in_baselink = np.zeros(2)
            #lin_in_baselink[0] = np.cos(yaw) * lin[0] - np.sin(yaw) * lin[1]
            #in_in_baselink[1] = np.sin(yaw) * lin[0] + np.cos(yaw) * lin[1]

        #lin_in_baselink = rot.T.dot(np.array([lin[0], lin[1], 0, 1]))[:2]

        #self.obs["sensor"][0,22] = lin_in_baselink[0]
        #self.obs["sensor"][0,23] = lin_in_baselink[1]
        #self.obs["sensor"][0,24] = 0#ang[0]
        #self.obs["sensor"][0,25] = 0#ang[1]
        #print(lin,alg)
   
    def sync_callback(self, rgb_msg, depth_msg):
        self.rgb_callback(rgb_msg)
        self.depth_callback(depth_msg)
        self.update_goal_and_vel()

    def update_goal_and_vel(self):
        if not self.goal_ready:
            return

        self.tf_listener_.waitForTransform("/base_link", "/map", rospy.Time(), rospy.Duration(4.0))
        position, quaternion = None, None
        data_populated = False
        while not data_populated:
            try:
                now = rospy.Time.now()
                self.tf_listener_.waitForTransform("/base_link", "/map", now, rospy.Duration(4.0))
                position, quaternion = self.tf_listener_.lookupTransform("/base_link", "/map", now)
                mat44 = self.fromTranslationRotation(position, quaternion)
                rotmat44 = self.fromTranslationRotation((0,0,0), quaternion)
                goal_xy_in_base_link_frame = np.dot(mat44, np.append(self.goal, 1.0))[:2]
                (lin_vel, ang_vel) = self.tf_listener_.lookupTwistFull('/base_link', '/map', '/base_link', (0,0,0), '/map', rospy.Time(0.0), rospy.Duration(0.5))
                lin_vel = rotmat44.dot(np.append(lin_vel, 1.0))[0]
                ang_vel = ang_vel[2]
                self.obs['sensor'][0:2] = goal_xy_in_base_link_frame
                self.obs['sensor'][2] = lin_vel
                self.obs['sensor'][3] = ang_vel
                data_populated = True
                
                
            except:
                print("waiting for tf")
                time.sleep(0.5)

        self.sensor_ready = True

    def rgb_callback(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        image = cv2.resize(image, (self.image_width, self.image_height))
        image = image.astype(np.float32) / 255.0
        #print('rgb', image.shape, np.mean(image), np.min(image), np.max(image))
        self.obs['rgb'] = image   
        #np_arr = np.fromstring(msg.data, np.uint8)
        #image_np = np_arr.reshape(480,640,3)

 
    def depth_callback(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
        image = cv2.resize(image, (self.image_width, self.image_height))
        image = image.astype(np.float32) / 1000.0
        image[image != image] = 0.0
        image[image < self.depth_min] = 0.0 
        image[image > self.depth_max] = 0.0
        image /= self.depth_max
        image = image[:, :, None]
        #print('depth', image.shape, np.mean(image), np.min(image), np.max(image))
        self.obs['depth'] = image
        #np_arr = np.fromstring(msg.data, np.uint16)
        #print(np_arr)
        #print(len(np_arr))
        #print((msg.header))
        #np_arr = msg.data            
        #image_np = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        #image_np = np_arr.reshape(640,480)
        #depth_img_mm = image_np

        #depth_img_mm = cv2.resize(depth_img_mm, (128, 96))
        #depth_img_m = depth_img_mm.astype(np.float32) / 1000.0
        #depth_img_m[depth_img_m > 4] = 4
        #depth_img_m[depth_img_m < 0.6] = -1

        #self.obs['depth'] = depth_img_m[None, :, :, None]
        # print(time.time() - self.last_time_depth_update)
        #self.last_time_depth_update = time.time()
        #print(np.max(depth_img_m), np.min(depth_img_m), np.mean(depth_img_m), depth_img_m.shape)
        #(lin, ang) = self.tf_listener_.lookupTwistFull('/base_link', '/map', '/base_link', (0,0,0), '/map', rospy.Time(0.0), rospy.Duration(0.5))
        #t = self.tf_listener_.getLatestCommonTime("/base_link", "/map")
        #position, quaternion = self.tf_listener_.lookupTransform("/base_link", "/map", t)
            
        #print(lin, ang, position, quaternion)
    
    def should_publish_action(self):
        return self.goal_ready and self.sensor_ready

    def submit(self, agent):
        while not rospy.is_shutdown():
            if not self.should_publish_action():
                rospy.sleep(rospy.Duration(0.2))
                continue
            action = agent.act(self.obs)
            vel_msg = Twist()
            vel_msg.linear.x = action[0] * 0.5
            vel_msg.angular.z = action[1] * np.pi / 2.0
            self.velocity_publisher.publish(vel_msg)
            print('action {}'.format(action))
            rospy.sleep(rospy.Duration(0.1))
