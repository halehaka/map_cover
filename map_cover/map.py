import numpy
import cv2
import math
from map_cover.grid import GridCellData, DiscreteProbabilityDistribution
import shapely

default_palette = numpy.array([[0, 0, 0],  # black - Background
                        [255, 0, 0],  # red - building
                        [0, 255, 0],  # green - woodland
                        [0, 0, 255],  # blue - water
                        [255, 255, 255], # white - road
                        [255, 255, 0], # Polygon 1
                        [255, 0, 255], # Polygon 2
                        [0, 255, 255], # Polygon 3
                        [127, 127, 127],# Polygon 4
                        ])  

"""class that represents a map with coordinates, including some helpful functions
"""
class Map:
    """ Initialize map at given coordinates (left_x : right_x, bottom_y : top_y)
    image can be a filename, in which case the file is read, or an image (cv2 or numpy array)
    """
    def __init__(self, left_x, right_x, bottom_y, top_y, image = None, flip_x = False, flip_y = True):
        assert(left_x < right_x)
        assert(bottom_y < top_y)
        self._left_x = left_x
        self._right_x = right_x
        self._bottom_y = bottom_y
        self._top_y = top_y
        if isinstance(image, str):
            self._image = cv2.imread(image)
        else:
            self._image = image
        
        self.flip_x = flip_x
        self.flip_y = flip_y

    @property
    def left_x(self):
        return self._left_x
    
    @property
    def right_x(self):
        return self._right_x

    @property
    def bottom_y(self):
        return self._bottom_y

    @property
    def top_y(self):
        return self._top_y        

    @property
    def size_x(self):
        return self.right_x - self.left_x
    
    @property
    def size_y(self):
        return self.top_y - self.bottom_y
    
    @property
    def image(self):
        return self._image
    
    @property
    def image_max_x(self):
        return self.image.shape[1] - 1
    
    @property
    def image_max_y(self):
        return self.image.shape[0] - 1
    
    def flip_coords(self,point):
        x,y = point
        if self.flip_x:
            rx = self.left_x + self.size_x - x
        else:
            rx = x
        if self.flip_y:
            ry = self.bottom_y + self.size_y - y
        else:
            ry = x
        return rx, ry

    def coords_to_relative_position(self, point):
        x,y = self.flip_coords(point)
        rx = (x - self.left_x) / self.size_x
        ry = (y - self.bottom_y) / self.size_y
        return rx, ry

    def relative_position_to_pixel(self, rpoint):
        rx, ry = rpoint
        x = int(round(self.image_max_x * rx))
        y = int(round(self.image_max_y * ry))        
        return (x,y)
    
    def coords_to_pixel(self, point):
        return self.relative_position_to_pixel(self.coords_to_relative_position(point))
    
    def pixel_to_relative_position(self, pixel_point):
        px, py = pixel_point
        x = px / self.size_x
        y = py / self.size_y
        return (x,y)
    
    def relative_position_to_coords(self, rpoint):
        x, y = rpoint
        cx = x * self.size_x + self.left_x
        cy = y * self.size_y + self.bottom_y
        return self.flip_coords( (cx, cy) )

    def pixel_to_coords(self, pixel_point):
        return self.relative_position_to_coords(self.pixel_to_relative_position(pixel_point))
    

    # A polygon is a list of coordinates of its vertices, e.g., [(1,2), (2,3), (4,5)]
    def color_polygon(self, polygon, polygon_id):
        polyshape = numpy.array(list(map(lambda x: numpy.array(self.coords_to_pixel(x)), polygon)))
        polyshape = polyshape.reshape((-1, 1, 2))
        cv2.fillPoly(self.image, [polyshape], polygon_id)

    def colormap(self, palette = default_palette):
        output_image = palette[numpy.int16(self.image)]
        return output_image
    
    def to_grid(self, delta_x, delta_y, cover_distributions):
        imax = self.image.max()
        bins = [x-0.5 for x in range(imax+2)]
        grid_x = numpy.arange(self.left_x, self.right_x, step=delta_x)
        grid_y = numpy.arange(self.bottom_y, self.top_y, step=delta_y)    
        grid = {}
        for i, cx in enumerate(grid_x):
            for j, cy in enumerate(grid_y):
                pixel_point = self.coords_to_pixel( (cx, cy) )
                next_cx = min(cx + delta_x, self.right_x)
                next_cy = min(cy + delta_y, self.top_y)
                next_pixel_point = self.coords_to_pixel( (next_cx, next_cy) )
                

                patch_min_x = min(pixel_point[0], next_pixel_point[0])
                patch_max_x = max(pixel_point[0], next_pixel_point[0])
                patch_min_y = min(pixel_point[1], next_pixel_point[1])
                patch_max_y = max(pixel_point[1], next_pixel_point[1])

                patch = self.image[patch_min_y:patch_max_y, patch_min_x:patch_max_x]
                shape = shapely.Polygon([(cx, cy), (cx, next_cy), (next_cx, next_cy), (next_cx, cy)])                
                histogram = numpy.histogram(patch, bins=bins, density=True)[0]
                
                # Compute weighted probability from histogram
                cell_distribution = DiscreteProbabilityDistribution(cover_distributions, histogram)
                grid[i,j] = GridCellData(shape, cell_distribution)

        return grid
    
        

    
