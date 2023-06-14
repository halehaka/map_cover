import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
import cv2
import cv_bridge

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.declare_parameter('filename', 'kaka_input.tif')
        self.publisher_ = self.create_publisher(Image, 'map_cover', 10)
        timer_period = 3  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        filename = self.get_parameter('filename').value
        cv_image = cv2.imread(filename)
        bridge = cv_bridge.CvBridge()
        
        msg = bridge.cv2_to_imgmsg(cv_image, "bgr8")        
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()