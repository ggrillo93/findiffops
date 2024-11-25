from base_op_class import *
from findiff import FinDiff
        
class ScalarDerOp(FinDiffOp):
    def __init__(self, N, xDer, yDer, dx = None, dy = None, BC = 'constant', bVal = None):
        """ BC can be constant (Dirichlet) or backward. Want to implement extrapolation and Pade scheme as well """

        assert BC == 'constant' or BC == 'backward'
        assert xDer + yDer <= 2
        assert xDer <=2 and yDer <= 2
        assert not xDer > 0 and dx is None
        assert not yDer > 0 and dy is None
        assert not BC == 'constant' and bVal is None

        if BC == 'backward':
            mat = self._FinDiffInit(N, xDer, yDer, dx, dy)
        else:
            mat = self._manualInit(N, xDer, yDer, dx, dy)

        super().__init__(mat)
        self.N, self.xDer, self.yDer, self.dx, self.dy, self.BC, self.bVal = N, xDer, yDer, dx, dy, BC, bVal

    def _FinDiffInit(self, N, xDer, yDer, dx, dy):
        if dx is None:
            mat = FinDiff(0, dy, yDer).matrix([N, N])
        elif dy is None:
            mat = FinDiff(1, dx, xDer).matrix([N, N])
        else:
            mat = FinDiff((1, dx, xDer), (0, dy, yDer)).matrix([N, N])
        return mat
    
    def _manualInit(self, N, xDer, yDer, dx, dy):
        I = np.eye(N)
        if dx is None:
            mat1D = FinDiff(0, dy, yDer).matrix((N,))
            mat1D[0] = mat1D[-1] = 0
            return kron(mat1D, I)
        elif dy is None:
            mat1D = FinDiff(0, dx, xDer).matrix((N,))
            mat1D[0] = mat1D[-1] = 0
            return kron(I, mat1D)
        else:
            mat1Dx = FinDiff(0, dx, xDer).matrix((N,))
            mat1Dx[0] = mat1Dx[-1] = 0
            mat1Dy = FinDiff(0, dy, yDer).matrix((N,))
            mat1Dy[0] = mat1Dy[-1] = 0
            return kron(mat1Dy, mat1Dx)

class VectorDerOp(FinDiffOp):
    pass