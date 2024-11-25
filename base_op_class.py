import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import *
from matplotlib.pyplot import spy, show

class FinDiffOp:
    def __init__(self, matrix):
        assert isinstance(matrix, spmatrix) or isinstance(matrix, np.ndarray)
        self.matrix = spmatrix(matrix)

    def apply(self, grid, bVal = None):
        soln = (self.matrix @ grid.flatten()).reshape(grid.shape)
        if bVal is None:
            return soln
        else:
            if isinstance(bVal, int, float, np.integer, np.floating):
                soln[0] = soln[-1] = bVal
                soln[:, 0] = soln[:, -1] = bVal
                return soln
            return NotImplemented
    
    def getMatrix(self):
        return np.copy(self.matrix)
    
    def __add__(self, obj):
        if isinstance(obj, FinDiffOp):
            return FinDiffOp(self.getMatrix() + obj.getMatrix())
        return NotImplemented
        
    def __radd__(self, obj):
        return self.__add__(obj)
    
    def __sub__(self, obj):
        if isinstance(obj, FinDiffOp):
            return FinDiffOp(self.getMatrix() - obj.getMatrix())
        return NotImplemented
    
    def __mul__(self, obj):
        if isinstance(obj, FinDiffOp):
            return FinDiffOp(self.getMatrix() @ obj.getMatrix())
        return NotImplemented
    
    def __rmul__(self, obj):
        return self.__mul__(obj)
    
    def visualize(self):
        spy(self.matrix)
        show()
        return
    
    def copy(self):
        return FinDiffOp(self.matrix)