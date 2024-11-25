from base_op_class import *
from findiff import FinDiff, Coefficient
        
class ScalarDerOp(FinDiffOp):
    def __init__(self, N, sequence, op1 = 1, op2 = 1, c1 = 1, c2 = 1, c3 = 1, BC = 'constant'):
        """ Returns matrix operator for c1 * (op1 * c2 * op2) * c3 up to second order derivatives (xDer + yDer <= 2).
            The coefficients are assumed to be 1D.
            Sequence can be ('R', 'R'), ('Z', 'Z'), ('R', 'Z'), or ('Z', 'R')
            BC can be constant (Dirichlet) or backward (takes the backward derivative)
            TODO: - Implement Pade scheme and extrapolation BCs
                  - Implement 2D coefficients
                  - Implement higher order accuracy
        """

        assert (BC == 'constant' or BC == 'backward')
        assert sequence == ('R', 'R') or sequence == ('R', 'Z') or sequence == ('Z', 'Z') or sequence == ('Z', 'R')

        if BC == 'backward':
            mat = self._FinDiffInit(N, op1, op2, c1, c2, c3)
        else:
            mat = self._manualInit(N, op1, op2, c1, c2, c3, sequence)
            
        super().__init__(mat)
        self.N, self.BC, self.sequence = N, BC, sequence

    def _FinDiffInit(self, N, op1, op2, c1, c2, c3):
        # 11.25.24 need to test this
        C1 = Coefficient(c1)
        C2 = Coefficient(c2)
        if isinstance(c3, (int, float, np.integer, np.floating)):
            C3 = np.eye(N) * c3
        elif c3.ndim == 1:
            C3 = diags(np.tile(c3, N), 0)
        mat = (C1 * op1 * (C2 * op2)).matrix([N, N]) @ C3
        return mat
    
    def _manualInit(self, N, op1, op2, c1, c2, c3, sequence):
        I = np.eye(N)
        C1 = Coefficient(c1)
        C2 = Coefficient(c2)
        C3 = I * c3 if isinstance(c3, (int, float, np.integer, np.floating)) else np.diag(c3)
        if sequence == ('R', 'R') or sequence == ('Z', 'Z'):
            mat1D = (C1 * op1 * (C2 * op2)).matrix((N, )) @ C3
            mat1D[0] = mat1D[-1] = 0
            if sequence[0] == 'Z':
                return kron(mat1D, I)
            else:
                return kron(I, mat1D)
        else:
            if sequence == ('R', 'Z'):
                mat1DR = (C1 * op1).matrix((N,)).toarray()
                mat1DZ = (C2 * op2).matrix((N,)).toarray() @ C3
            else:
                mat1DR = (C2 * op2).matrix((N,)).toarray() @ C3
                mat1DZ = (C1 * op1).matrix((N,)).toarray()
            mat1DR[0] = mat1DZ[0] = mat1DR[-1] = mat1DZ[-1] = 0
            return kron(mat1DZ, mat1DR)
        
class BasicDerOp(ScalarDerOp):
    def __init__(self, N, RDer, ZDer, dR = None, dZ = None, BC = 'constant'):
        assert (RDer + ZDer <= 2)
        assert not (RDer > 0 and dR is None)
        assert not (ZDer > 0 and dZ is None)

        if dZ is None:
            dROp = FinDiff(0, dR, RDer)
            super().__init__(N, ('R', 'R'), op1 = dROp, BC = BC)
        elif dR is None:
            dZOp = FinDiff(0, dZ, ZDer)
            super().__init__(N, ('Z', 'Z'), op1 = dZOp, BC = BC)
        else:
            dROp = FinDiff(0, dR)
            dZOp = FinDiff(0, dZ)
            super().__init__(N, ('R', 'Z'), op1 = dROp, op2 = dZOp, BC = BC)
    
class ScalarLapOp(FinDiffOp):
    def __init__(self, N, dR, dZ, R, BC = 'constant'):
        dROp = FinDiff(0, dR)
        dRMat = ScalarDerOp(N, ('R', 'R'), op1 = dROp, op2 = dROp, c1 = 1 / R, c2 = R, BC = BC)
        dZMat = BasicDerOp(N, 0, 2, dZ = dZ, BC = BC)
        super().__init__(dRMat + dZMat)

class VectorLaplacianOp(FinDiffOp):
    def __init__(self, N, dR, dZ, R, BC = 'constant'):
        scLap = ScalarLapOp(N, dR, dZ, R, BC = BC)
        modEye = np.eye(3)
        modEye[2, 2] = 0
        mat = kron(scLap, eye(3)) - kron(kron(np.diag(1 / R ** 2), eye(N)), eye(N))
        super().__init__(mat)

class GradOp(FinDiffOp):
    pass

class DivOp(FinDiffOp):
    pass

class GradDivOp(FinDiffOp):
    pass

class CurlOp(FinDiffOp):
    """ Returns the curl operator in axisymmetric cylindrical coordinates """
    def __init__(self, N, dR, dZ, R, BC = 'constant'):
        dROp = FinDiff(0, dR)
        dRMat0 = BasicDerOp(N, 1, 0, dR = dR, BC = BC).getMatrix()
        dRMat1 = ScalarDerOp(N, ('R', 'R'), op2 = dROp, c2 = 1 / R, c3 = R, BC = BC).getMatrix()
        dZMat = BasicDerOp(N, 0, 1, dZ = dZ, BC = BC).getMatrix()
        mat = self._init_with_kron(dRMat0, dRMat1, dZMat)
        super().__init__(mat)
        self.N, self.dR, self.dZ, self.R, self.BC = N, dR, dZ, R, BC
    
    def _init_with_kron(self, dRMat0, dRMat1, dZMat):
        MZ = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        opZ = kron(dZMat, MZ)

        MR0 = np.array([[0, 0, 0], [0, 0, -1], [0, 0, 0]])
        MR1 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])

        opR = kron(dRMat0, MR0) + kron(dRMat1, MR1)
        return opZ + opR