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
        mat = kron(diags(AR.flatten(), 0), RMat) + kron(diags(Aphi.flatten(), 0), phiMat) + kron(diags(AZ.flatten(), 0), ZMat)
        return mat, N
    
    def apply(self, grid):
        assert grid.shape == (self.N, self.N, 3)
        soln = (self.matrix @ grid.flatten()).reshape(self.N, self.N, 3)
        return soln
    
class MatOp(FinDiffOp):
    def __init__(self, matGrid):
        assert matGrid.ndim == 4
        mat = self._init_with_kron(matGrid)
        super().__init__(mat)
        self.N = matGrid.shape[0]

    def _init_with_kron(self, matGrid):
        MRR, MRphi, MRZ = [matGrid[:, :, 0, i] for i in range(3)]
        MphiR, Mphiphi, MphiZ = [matGrid[:, :, 1, i] for i in range(3)]
        MZR, MZphi, MZZ = [matGrid[:, :, 2, i] for i in range(3)]
        unitMats = np.zeros([9, 3, 3])
        unitMats[0, 0, 0] = unitMats[1, 0, 1] = unitMats[2, 0, 2] = 1
        unitMats[3, 1, 0] = unitMats[4, 1, 1] = unitMats[5, 1, 2] = 1
        unitMats[6, 2, 0] = unitMats[7, 2, 1] = unitMats[8, 2, 2] = 1
        MFlat = [MRR, MRphi, MRZ, MphiR, Mphiphi, MphiZ, MZR, MZphi, MZZ]
        mat = 0
        for i in range(9):
            mat += kron(diags(MFlat[i].flatten(), 0), unitMats[i])
        return mat