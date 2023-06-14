import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
import predict
import cv2
import cv_bridge


model_name = "../data/v7-LandCover-retrained-twice"

class MapCoverNode(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.model = predict.load_model(model_name)
        self.bridge = cv_bridge.CvBridge()
        self.subscription = self.create_subscription(
            Image,
            'map_cover',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('got message')
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        output = predict.handle_image(self.model, cv_image)


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