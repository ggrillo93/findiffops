import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import *
from matplotlib.pyplot import spy, show
from minresQLP import MinresQLP

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
        mat = self.getMatrix().tolil()
        for idx in bInd:
            mat.rows[idx] = [idx]
            mat.data[idx] = [1.0]
        return mat.tocsr()
    
    def visualize(self):
        spy(self.matrix)
        show()
        return
    
    def copy(self):
        return FinDiffOp(self.matrix)
    
    def invert_simple(self, S, BC):
        BCMat = self.enforceBC(S.shape)
        S[:, 0], S[:, -1], S[0], S[-1] = BC
        return spsolve(BCMat, S.flatten()).reshape(S.shape)
    
    def invert_minresQLP(self, S, BC, rtol = 1e-7, maxit = 1e3, debug = False):
        BCMat = self.enforceBC(S.shape)
        S[:, 0], S[:, -1], S[0], S[-1] = BC
        soln = MinresQLP(BCMat, S.flatten(), rtol, maxit)
        if debug:
            return soln
        else:
            return soln[0].reshape(S.shape)