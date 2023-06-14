import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
import map_cover.predict
import cv2
import cv_bridge

from ament_index_python.packages import get_package_share_directory
import os
# may raise PackageNotFoundError
package_share_directory = get_package_share_directory('map_cover')

class MapCoverNode(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.declare_parameter('model_name', os.path.join(get_package_share_directory('map_cover'), 'resource', 'v7-LandCover-retrained-twice'))
        model_name = self.get_parameter('model_name').value
        self.model = map_cover.predict.load_model(model_name)
        self.bridge = cv_bridge.CvBridge()
        self.subscription = self.create_subscription(
            Image,
            'map_cover',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.map_cover_publisher = self.create_publisher(Image, "map_cover_labels", 10)  

    def listener_callback(self, msg):
        self.get_logger().info('got message')
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        output, output_image = map_cover.predict.handle_image(self.model, cv_image)   
        cv2.imwrite("kaka.jpeg", output_image)
        image2 = cv2.imread("kaka.jpeg")     
        self.map_cover_publisher.publish(self.bridge.cv2_to_imgmsg(image2, "bgr8"))


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MapCoverNode()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()