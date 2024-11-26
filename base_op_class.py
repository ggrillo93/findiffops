import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import *
from matplotlib.pyplot import spy, show

class FinDiffOp:
    def __init__(self, matrix):
        assert isinstance(matrix, spmatrix) or isinstance(matrix, np.ndarray)
        self.matrix = matrix

    def apply(self, grid, bVals = None):
        soln = (self.matrix @ grid.flatten()).reshape(grid.shape)
        if bVals is None:
            return soln
        else:
            lrVals, tbVals = bVals
            if lrVals is not None:
                soln[:, 0], soln[:, -1] = lrVals
            if tbVals is not None:
                soln[0], soln[-1] = tbVals
            return soln
    
    def getMatrix(self):
        return self.matrix
    
    def setMatrix(self, mat):
        self.matrix = mat
        return
    
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
    
    def enforceBC(self, shape):
        ones = np.ones(shape)
        ones[1:-1, 1:-1] = 0
        bInd = np.nonzero(ones.flatten())[0]
        mat = self.getMatrix.tolil()
        mat[bInd, bInd] = 1
        self.setMatrix(mat.tocsr())
        return
    
    def visualize(self):
        spy(self.matrix)
        show()
        return
    
    def copy(self):
        return FinDiffOp(self.matrix)