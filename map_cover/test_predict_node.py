import rclpy
from rclpy.node import Node
import sys

from swarm_interfaces.srv import MapToGrid
from swarm_interfaces.msg import MapWithCoords, Zone, GeoPoint, ZoneShapeEnum, ZoneTypeEnum, MissionTargetData


def geopoint_from_xy(x, y):
    return GeoPoint(latitude = float(x), longitude = float(y), altitude = 0.0)

class MapToGridClientAsync(Node):
    def __init__(self):
        super().__init__('MapToGrid_client')
        self.cli = self.create_client(MapToGrid, 'map_to_grid')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = MapToGrid.Request()

    def send_request(self, filename):
        self.req.input_map.map_filename = filename
        self.req.input_map.bottom_left = geopoint_from_xy(0, 0)
        self.req.input_map.top_right = geopoint_from_xy(100, 100)

        polygon1 = Zone()
        polygon1.geo_points = [geopoint_from_xy(5,5), geopoint_from_xy(15, 15), geopoint_from_xy(15, 5)]
        polygon1.target_data.target_priority = [1,5]
        polygon1.target_data.target_detection_probability = [0.8, 0.2]

        polygon2 = Zone()
        polygon2.geo_points = [geopoint_from_xy(60, 60), geopoint_from_xy(60, 90), geopoint_from_xy(90, 90), geopoint_from_xy(90, 60)]
        polygon2.target_data.target_priority = [1,5, 10]
        polygon2.target_data.target_detection_probability = [0.2, 0.3, 0.5]

        self.req.additional_polygons = [polygon1, polygon2]

        self.req.grid_cell_size = 10.0

        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()



def main(args=None):
    rclpy.init(args=args)

    minimal_client = MapToGridClientAsync()
    response = minimal_client.send_request(sys.argv[1])
    minimal_client.get_logger().info(
        'Result:  ' +  str(response))        

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()