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
        elif isinstance(int, np.integer, float, np.floating):
            soln[:, 0], soln[:, -1] = soln[0] = soln[-1] = bVals
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
    
    def findBInd(self, shape):
        ones = np.ones(shape)
        ones[1:-1, 1:-1] = 0
        bInd = np.nonzero(ones.flatten())[0]
        return bInd
    
    def enforceBC(self, shape):
        bInd = self.findBInd(shape)
        mat = self.getMatrix().tolil()
        for idx in bInd:
            mat.rows[idx] = [idx]
            mat.data[idx] = [1.0]
        return mat.tocsr()
    
    def getInteriorMat(self, shape):
        bInd = self.findBInd(shape)
        all_idx = np.arange(np.prod(shape))
        interior_idx = np.setdiff1d(all_idx, bInd)
        mat = self.getMatrix()
        return mat[interior_idx, :][:, interior_idx], mat[interior_idx, :][:, bInd]
    
    def getCompMats(self):
        mat = self.getMatrix()
        n = mat.shape[0]
        R_idx = np.arange(0, n, 3)
        phi_idx = np.arange(n)[1::3]
        Z_idx = np.arange(2, n, 3)
        O_R = mat[R_idx, :]
        O_phi = mat[phi_idx, :]
        O_Z = mat[Z_idx, :]
        return O_R, O_phi, O_Z
    
    def visualize(self):
        spy(self.matrix)
        show()
        return
    
    def copy(self):
        return FinDiffOp(self.matrix)
    
    def invert_simple(self, S, BC, BCEnf = 'direct'):
        assert BCEnf == 'direct' or BCEnf == 'adjust'
        S[:, 0], S[:, -1], S[0], S[-1] = BC
        if BCEnf == 'direct':
            BCMat = self.enforceBC(S.shape)
            return spsolve(BCMat, S.flatten()).reshape(S.shape)
        else:
            bInd = self.findBInd(S.shape)
            intMat, adjMat = self.getInteriorMat(S.shape)
            SInt = S[1:-1, 1:-1]
            RHS = SInt.flatten() - adjMat.dot(S.flatten()[bInd])
            solnInt = spsolve(intMat, RHS).reshape(SInt.shape)
            solnAll = np.copy(S)
            solnAll[1:-1, 1:-1] = solnInt
            return solnAll
    
    def invert_minresQLP(self, S, BC, rtol = 1e-7, maxit = 1e3, debug = False):
        BCMat = self.enforceBC(S.shape)
        S[:, 0], S[:, -1], S[0], S[-1] = BC
        soln = MinresQLP(BCMat, S.flatten(), rtol, maxit)
        if debug:
            return soln
        else:
            return soln[0].reshape(S.shape)