from typing import List
import numbers
import shapely

"""class that represents the data associated with a grid cell
"""
class DiscreteProbabilityDistribution:
    """ values can be a dictionary mapping values to probabilities or 
    a list of DiscreteProbabilityDistribution, in which case we assume the second parameter is list of weights of the same lengths
    """
    def __init__(self, values, weights = None):        
        if weights is None:            
            self.data = values.copy()
        else:
            assert(len(values) == len(weights))
            assert(isinstance(values[0], DiscreteProbabilityDistribution))
            joint_values = set()
            for d in values.values():
                joint_values = joint_values.union(d.values)
            self.data = {}
            for val in joint_values:
                self.data[val] = sum([values[i][val] * weights[i] for i in range(len(values))])


    def __getitem__(self, value):
        return self.data.get(value, 0)
    
    def __setitem__(self, value, prob):
        self.data[value] = prob        

    def __repr__(self):
        return "DiscreteProbabilityDistribution: " + str(self.data)

    @property
    def values(self):
        return self.data.keys()    

    def __add__(self, other):
        assert(isinstance(other,DiscreteProbabilityDistribution))
        joint_values = sorted(list(set(self.values).union(set(other.values))))
        return DiscreteProbabilityDistribution(joint_values, list(map(lambda val: self[val] + other[val], joint_values)))
    
    def __mul__(self, other):
        assert(isinstance(other, numbers.Real))
        return DiscreteProbabilityDistribution(self.values, list(map(lambda val: self[val] * other, self.values)))


class GridCellData:
    def __init__(self, area : shapely.Geometry, target_data : DiscreteProbabilityDistribution, blocked : bool = None):
        self._area = area
        self._target_data = target_data
        self._blocked = False        

    @property
    def blocked(self):
        return self._blocked
    
    @blocked.setter
    def blocked(self, value):
        self._blocked = value

    @property
    def target_data(self):
        return self._target_data

    @property
    def area(self):
        return self._area

    def __repr__(self):
        return "area:" + str(self.area) + "__blocked:" + str(self.blocked) + "__target:" + str(self.target_data)
