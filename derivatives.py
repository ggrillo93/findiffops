from base_op_class import *
from findiff import FinDiff, Coefficient
        
class ScalarDerOp(FinDiffOp):
    def __init__(self, N, op1 = 1, op2 = 1, c1 = 1, c2 = 1, c3 = 1):
        """ Returns matrix operator for c1 * (op1 * c2 * op2) * c3 up to second order derivatives (xDer + yDer <= 2).
            The coefficients are assumed to be 1D.
            Sequence can be ('R', 'R'), ('Z', 'Z'), ('R', 'Z'), or ('Z', 'R')
            BC can be constant (Dirichlet) or backward (takes the backward derivative)
            TODO: - Implement Pade scheme and extrapolation BCs
                  - Implement 2D coefficients
                  - Implement higher order accuracy
        """

        mat = self._FinDiffInit(N, op1, op2, c1, c2, c3)
            
        super().__init__(mat)
        self.N = N

    def _FinDiffInit(self, N, op1, op2, c1, c2, c3):
        C1 = Coefficient(c1)
        C2 = Coefficient(c2)
        if isinstance(c3, (int, float, np.integer, np.floating)):
            mat = (C1 * op1 * (C2 * op2)).matrix([N, N]) * c3
        else:
            mat = (C1 * op1 * (C2 * op2)).matrix([N, N]) @ diags(c3.flatten(), 0)
        return mat
        
class BasicDerOp(ScalarDerOp):
    def __init__(self, N, RDer, ZDer, dR = None, dZ = None):
        assert (RDer + ZDer <= 2)
        assert not (RDer > 0 and dR is None)
        assert not (ZDer > 0 and dZ is None)

        if dZ is None:
            dROp = FinDiff(1, dR, RDer)
            super().__init__(N, op1 = dROp)
        elif dR is None:
            dZOp = FinDiff(0, dZ, ZDer)
            super().__init__(N, op1 = dZOp)
        else:
            dROp = FinDiff(1, dR)
            dZOp = FinDiff(0, dZ)
            super().__init__(N, op1 = dROp, op2 = dZOp)
    
class ScalarLapOp(FinDiffOp):
    def __init__(self, N, dR, dZ, R):
        dROp = FinDiff(1, dR)
        dRMat = ScalarDerOp(N, op1 = dROp, op2 = dROp, c1 = 1 / R, c2 = R).getMatrix()
        dZMat = BasicDerOp(N, 0, 2, dZ = dZ).getMatrix()
        mat = dRMat + dZMat
        super().__init__(mat)
        self.N, self.dR, self.dZ, self.R = N, dR, dZ, R

class VecLapOp(FinDiffOp):
    def __init__(self, N, dR, dZ, R):
        scLap = ScalarLapOp(N, dR, dZ, R).getMatrix()
        modEye = np.eye(3)
        modEye[2, 2] = 0
        mat = kron(scLap, eye(3)) - kron(diags(1 / (R.flatten() ** 2), 0), modEye)
        super().__init__(mat)
        self.N, self.dR, self.dZ, self.R = N, dR, dZ, R

class DirDerivOp(FinDiffOp):
    def __init__(self, N, dR, dZ, R, A):
        assert A.shape == (N, N, 3)
        AR, Aphi, AZ = [A[..., i] for i in range(3)]
        ddR = FinDiff(1, dR)
        ddZ = FinDiff(0, dZ)
        ARCoeff, AZCoeff = Coefficient(AR), Coefficient(AZ)
        mat = kron((ARCoeff * ddR + AZCoeff * ddZ).matrix([N, N]), eye(3))
        multMat = np.zeros([3, 3])
        multMat[0, 1] = -1
        multMat[1, 0] = 1
        mat += kron(diags((Aphi / R).flatten(), 0), multMat)
        super().__init__(mat)
        self.N, self.dR, self.dZ, self.R = N, dR, dZ, R

class GradDivOpWCoeff(FinDiffOp):
    """ Returns operator corresponding to grad(a div(MA)), with a a scalar, M a matrix, and A a vector. The operator acts on A """
    def __init__(self, N, dR, dZ, R, scGrid, matGrid):
        mat = self._init_with_kron(N, dR, dZ, R, scGrid, matGrid)
        super().__init__(mat)
        self.N, self.dR, self.dZ, self.R = N, dR, dZ, R
    
    def _init_with_kron(self, N, dR, dZ, R, scGrid, matGrid):
        # Matrix indices are not right
        ARR, AphiR, AZR = [matGrid[:, :, 0, i] for i in range(3)]
        ARZ, AphiZ, AZZ = [matGrid[:, :, 2, i] for i in range(3)]
        kronMats = np.zeros([6, 3, 3])
        kronMats[0, 0, 0], kronMats[1, 0, 1], kronMats[2, 0, 2] = np.ones(3)
        kronMats[3, 2, 0], kronMats[4, 2, 1], kronMats[5, 2, 2] = np.ones(3)
        ddR = FinDiff(1, dR)
        ddZ = FinDiff(0, dZ)
        scGridOverR = scGrid / R
        MRR = ScalarDerOp(N, op1 = ddR, op2 = ddR, c2 = scGridOverR, c3 = R * ARR).getMatrix()
        MRR += ScalarDerOp(N, op1 = ddR, op2 = ddZ, c2 = scGrid, c3 = ARZ).getMatrix()
        MphiR = ScalarDerOp(N, op1 = ddR, op2 = ddR, c2 = scGridOverR, c3 = R * AphiR).getMatrix()
        MphiR += ScalarDerOp(N, op1 = ddR, op2 = ddZ, c2 = scGrid, c3 = AphiZ).getMatrix()
        MZR = ScalarDerOp(N, op1 = ddR, op2 = ddR, c2 = scGridOverR, c3 = R * AZR).getMatrix()
        MZR += ScalarDerOp(N, op1 = ddR, op2 = ddZ, c2 = scGrid, c3 = AZZ).getMatrix()
        MRZ = ScalarDerOp(N, op1 = ddZ, op2 = ddR, c2 = scGridOverR, c3 = R * ARR).getMatrix()
        MRZ += ScalarDerOp(N, op1 = ddZ, op2 = ddZ, c2 = scGrid, c3 = ARZ).getMatrix()
        MphiZ = ScalarDerOp(N, op1 = ddZ, op2 = ddR, c2 = scGridOverR, c3 = R * AphiR).getMatrix()
        MphiZ += ScalarDerOp(N, op1 = ddZ, op2 = ddZ, c2 = scGrid, c3 = AphiZ).getMatrix()
        MZZ = ScalarDerOp(N, op1 = ddZ, op2 = ddR, c2 = scGridOverR, c3 = R * AZR).getMatrix()
        MZZ += ScalarDerOp(N, op1 = ddZ, op2 = ddZ, c2 = scGrid, c3 = AZZ).getMatrix()
        mats = [MRR, MphiR, MZR, MRZ, MphiZ, MZZ]
        mat = 0
        for i in range(len(mats)):
            mat += kron(mats[i], kronMats[i])
        return mat

class ScGradOp(FinDiffOp):
    def __init__(self, N, dR, dZ):
        RMat = FinDiff(1, dR).matrix([N, N])
        ZMat = FinDiff(0, dZ).matrix([N, N])
        mat = kron(RMat, np.array([[1, 0, 0]]).T) + kron(ZMat, np.array([[0, 0, 1]]).T)
        super().__init__(mat)
        self.N, self.dR, self.dZ = N, dR, dZ

    def apply(self, grid, bVals = None):
        assert grid.shape == (self.N, self.N)
        soln = (self.matrix @ grid.flatten()).reshape(self.N, self.N, 3)
        if bVals is None:
            return soln
        else:
            lrVals, tbVals = bVals
            if lrVals is not None:
                soln[:, 0], soln[:, -1] = lrVals
            if tbVals is not None:
                soln[0], soln[-1] = tbVals
            return soln
        
class VecGradOp(FinDiffOp):
    def __init__(self, N, dR, dZ, R):
        RMat = kron(FinDiff(1, dR).matrix([N, N]), np.eye(3))
        ZMat = kron(FinDiff(0, dZ).matrix([N, N]), np.eye(3))
        midMat = kron(diags(1 / R.flatten(), 0), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
        mat = kron(RMat, np.array([[1, 0, 0]]).T) + kron(midMat, np.array([[0, 1, 0]]).T) + kron(ZMat, np.array([[0, 0, 1]]).T)
        super().__init__(mat)
        self.N, self.dR, self.dZ, self.R = N, dR, dZ, R
    
    def apply(self, grid, bVals = None):
        assert grid.shape == (self.N, self.N, 3)
        soln = (self.matrix @ grid.flatten()).reshape(self.N, self.N, 3, 3)
        if bVals is None:
            return soln
        else:
            lrVals, tbVals = bVals
            if lrVals is not None:
                soln[:, 0], soln[:, -1] = lrVals
            if tbVals is not None:
                soln[0], soln[-1] = tbVals
            return soln

class DivOp(FinDiffOp):
    def __init__(self, N, dR, dZ, R):
        dROp = FinDiff(1, dR)
        RMat = ScalarDerOp(N, op2 = dROp, c2 = 1 / R, c3 = R).getMatrix()
        ZMat = FinDiff(0, dZ).matrix([N, N])
        mat = kron(RMat, np.array([1, 0, 0])) + kron(ZMat, np.array([0, 0, 1]))
        super().__init__(mat)
        self.N, self.dR, self.dZ, self.R = N, dR, dZ, R

    def apply(self, grid, bVals = None):
        assert grid.shape == (self.N, self.N, 3)
        soln = (self.matrix @ grid.flatten()).reshape(self.N, self.N)
        if bVals is None:
            return soln
        else:
            lrVals, tbVals = bVals
            if lrVals is not None:
                soln[:, 0], soln[:, -1] = lrVals
            if tbVals is not None:
                soln[0], soln[-1] = tbVals
            return soln

class GradDivOp(FinDiffOp):
    def __init__(self, N, dR, dZ, R):
        mat = self._init_with_kron(N, dR, dZ, R)
        super().__init__(mat)
        self.N, self.dR, self.dZ, self.R = N, dR, dZ, R
    
    def _init_with_kron(self, N, dR, dZ, R):
        dROp = FinDiff(1, dR)
        dZOp = FinDiff(0, dZ)
        MRROp = ScalarDerOp(N, op1 = dROp, op2 = dROp, c2 = 1 / R, c3 = R).getMatrix()
        MRZOp = ScalarDerOp(N, op1 = dROp, op2 = dZOp, c1 = 1 / R, c2 = R).getMatrix()
        MZROp = BasicDerOp(N, 1, 1, dR = dR, dZ = dZ).getMatrix()
        MZZOp = BasicDerOp(N, 0, 2, dZ = dZ).getMatrix()
        MRRMat, MRZMat, MZRMat, MZZMat = np.zeros([4, 3, 3])
        MRRMat[0, 0] = MRZMat[2, 0] = MZRMat[0, 2] = MZZMat[2, 2] = 1
        mat = kron(MRROp, MRRMat) + kron(MRZOp, MRZMat) + kron(MZROp, MZRMat) + kron(MZZOp, MZZMat)
        return mat

class CurlCurlOp(FinDiffOp):
    def __init__(self, N, dR, dZ, R):
        GradDiv = GradDivOp(N, dR, dZ, R)
        Lap = VecLapOp(N, dR, dZ, R)
        super().__init__(GradDiv.getMatrix() - Lap.getMatrix())
        self.N, self.dR, self.dZ, self.R = N, dR, dZ, R

class CurlOp(FinDiffOp):
    """ Returns the curl operator in axisymmetric cylindrical coordinates """
    def __init__(self, N, dR, dZ, R):
        dROp = FinDiff(1, dR)
        dRMat0 = BasicDerOp(N, 1, 0, dR = dR).getMatrix()
        dRMat1 = ScalarDerOp(N, op2 = dROp, c2 = 1 / R, c3 = R).getMatrix()
        dZMat = BasicDerOp(N, 0, 1, dZ = dZ).getMatrix()
        mat = self._init_with_kron(dRMat0, dRMat1, dZMat)
        super().__init__(mat)
        self.N, self.dR, self.dZ, self.R = N, dR, dZ, R
    
    def _init_with_kron(self, dRMat0, dRMat1, dZMat):
        MZ = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        opZ = kron(dZMat, MZ)

        MR0 = np.array([[0, 0, 0], [0, 0, -1], [0, 0, 0]])
        MR1 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])

        opR = kron(dRMat0, MR0) + kron(dRMat1, MR1)
        return opZ + opR