import rclpy
from rclpy.node import Node
import map_cover.predict, map_cover.map, map_cover.grid
import cv2
from ament_index_python.packages import get_package_share_directory
import os
from swarm_interfaces.srv import MapToGrid
from swarm_interfaces.msg import MapWithCoords, Zone, GeoPoint, ZoneShapeEnum, ZoneTypeEnum


# may raise PackageNotFoundError
package_share_directory = get_package_share_directory('map_cover')

class MapToGridServiceNode(Node):

    def __init__(self):
        super().__init__('MapToGrid')
        
        self.declare_parameter('model_name', os.path.join(get_package_share_directory('map_cover'), 'resource', 'v7-LandCover-retrained-twice'))
        model_name = self.get_parameter('model_name').value
        self.model = map_cover.predict.load_model(model_name)

        self.declare_parameter('cover_distributions_filename', os.path.join(get_package_share_directory('map_cover'), 'resource', 'cover_distribution.yaml'))
        cover_distributions_filename = self.get_parameter('cover_distributions_filename').value
        self.cover_distributions = map_cover.predict.load_cover_distributions(cover_distributions_filename)
        
        self.srv = self.create_service(MapToGrid, 'map_to_grid', self.map_to_grid_callback)        

    def map_to_grid_callback(self, request, response):
        self.get_logger().info('got request: ' +  request.input_map.map_filename)

        # Load Map from request        
        map = map_cover.map.Map(request.input_map.bottom_left.longitude, request.input_map.top_right.longitude,
                                request.input_map.bottom_left.latitude, request.input_map.top_right.latitude,
                                request.input_map.map_filename)            
        self.get_logger().info('map loaded, predicting cover labels')

        pred = map_cover.predict.image_to_pixel_cover(self.model, map.image)
        map_pred = map_cover.map.Map(0, 100, 0, 100, pred)

        self.get_logger().info('cover labels predicted, adding polygons')
        cover_distributions = self.cover_distributions.copy()
        polygon_id = map_cover.predict.NUM_CLASSES
        for zone in request.additional_polygons:
            polygon = [(point.longitude, point.latitude) for point in zone.geo_points]
            dist = map_cover.grid.DiscreteProbabilityDistribution(
                {zone.target_data.target_priority[i] : zone.target_data.target_detection_probability[i] for i in range(len(zone.target_data.target_priority))})
            map_pred.color_polygon(polygon, polygon_id)   
            cover_distributions[polygon_id] = dist
            polygon_id = polygon_id + 1

        self.get_logger().info('added polygons, overlaying grid')
        grid = map_pred.to_grid(request.grid_cell_size, request.grid_cell_size, cover_distributions)

        for i,j in grid:            
            z = Zone()
            z.geo_points = [GeoPoint(longitude=x, latitude=y, altitude=0.0) for x,y in grid[i,j].area.exterior.coords]
            z.target_data.target_priority = grid[i,j].target_data.values
            z.target_data.target_detection_probability = [grid[i,j].target_data[val] for val in grid[i,j].target_data.values]
            z.zone_type.value = ZoneTypeEnum.TARGET_DATA_ZONE
            z.zone_shape.value = ZoneShapeEnum.POLYGON
            response.output_grid.append(z)
        return response


def main(args=None):
    rclpy.init(args=args)

    map_to_grid_node = MapToGridServiceNode()

    rclpy.spin(map_to_grid_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    map_to_grid_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()