from base_op_class import *

class DotOp(FinDiffOp):
    def __init__(self, vecGrid):
        assert vecGrid.ndim == 3
        N = vecGrid.shape[0]
        ops1D = []
        for i in range(N):
            op1D = np.zeros([N, N, 3])
            for j in range(N):
                op1D[j, j] = vecGrid[i, j]
            op1D = op1D.reshape(N, 3 * N)
            ops1D.append(op1D)
        mat = block_diag(ops1D, format = 'csr')
        super().__init__(mat)
        self.N = N
    
    def apply(self, grid):
        assert grid.shape == (self.N, self.N, 3)
        soln = (self.matrix @ grid.flatten()).reshape(self.N, self.N)
        return soln

class CrossOp(FinDiffOp):
    def __init__(self, vecGrid):
        assert vecGrid.ndim == 3
        mat, N = self._init_with_kron(vecGrid)
        super().__init__(mat)
        self.N = N

    def _init_with_kron(self, vecGrid):
        N = vecGrid.shape[0]
        AR, Aphi, AZ = [vecGrid[..., i] for i in range(3)]
        RMat, phiMat, ZMat = np.zeros([3, 3, 3])
        RMat[2, 1] = phiMat[0, 2] = ZMat[1, 0] = 1
        RMat[1, 2] = phiMat[2, 0] = ZMat[0, 1] = -1
        mat = kron(AR, RMat) + kron(Aphi, phiMat) + kron(AZ, ZMat)
        return mat, N