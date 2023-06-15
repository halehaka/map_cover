import numpy
import cv2
import math

palette = numpy.array([[0, 0, 0],  # black - Background
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
        x = int(round(self.image.shape[0] * rx))
        y = int(round(self.image.shape[1] * ry))
        #return numpy.array([x,y],dtype=numpy.int32)    
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
    

    def color_polygon(self, polygon, polygon_id):
        polyshape = numpy.array(list(map(lambda x: numpy.array(self.coords_to_pixel(x)), polygon)))
        polyshape = polyshape.reshape((-1, 1, 2))
        cv2.fillPoly(self.image, [polyshape], polygon_id)

    def colormap(self):
        output_image = palette[numpy.int16(self.image)]
        return output_image
    
    def to_grid(self, delta_x, delta_y):
        print(self.image.shape)
        delta_x_pixels = math.ceil(self.image.shape[1] / (self.size_x / delta_x))
        delta_y_pixels = math.ceil(self.image.shape[0] / (self.size_y / delta_y))
        
        grid = {}
        for j,y in enumerate(range(0,self.image.shape[0], delta_y_pixels)):
            for i,x in enumerate(range(0,self.image.shape[1], delta_x_pixels)):            
                next_x = min(x+delta_x_pixels, self.image.shape[1])
                next_y = min(y+delta_y_pixels, self.image.shape[0])
                patch = self.image[y:next_y, x:next_x]
                grid[i,j] = (self.pixel_to_coords( (x,y) ), self.pixel_to_coords( (next_x, next_y) ), numpy.histogram(patch, bins=[x-0.5 for x in range(self.image.max())], density=True)[0])
        return grid
    
        

    
