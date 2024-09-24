from __future__ import division
import torch
import random
import numpy as np
import numbers
import types

import scipy.ndimage as ndimage

class Compose(object):
    
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms
        
    def __call__(self, input, target):
        for t in self.co_transforms:
            input, target = t(input, target)
        return input, target

class ArrayToTensor(object):
    
    def __call__(self, array):
        assert(isinstance(array, np.ndarry))
        array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array)
        return tensor.float()
    
class Lambda(object):
    
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd
    
    def __call__(self, input, target):
        return self.lambd(input, target)
    