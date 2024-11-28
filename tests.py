from derivatives import *
from vector_ops import *
from numpy.testing import assert_array_almost_equal

def gradFD(scField, dR, dZ):
    ddR = FinDiff(1, dR)
    ddZ = FinDiff(0, dZ)
    solnR = ddR(scField)
    solnZ = ddZ(scField)
    return np.stack((solnR, np.zeros_like(solnR), solnZ), axis = -1)

def divFD(vecField, dR, dZ, RGrid):
    ddR = FinDiff(1, dR)
    ddZ = FinDiff(0, dZ)
    return ddR(RGrid * vecField[..., 0]) / RGrid + ddZ(vecField[..., 2])

def curlFD(A, dR, dZ, RGrid):
    ddR = FinDiff(1, dR)
    ddZ = FinDiff(0, dZ)
    AR, Aphi, AZ = [A[:, :, i] for i in range(3)]
    result = np.stack((-ddZ(Aphi), ddZ(AR) - ddR(AZ), ddR(RGrid * Aphi) / RGrid), axis = -1)
    return result

def scLapFD(scField, dR, dZ, RGrid):
    ddR = FinDiff(1, dR)
    ddZZ = FinDiff(0, dZ, 2)
    return ddR(RGrid * ddR(scField)) / RGrid + ddZZ(scField)

def vecLapFD(vecField, dR, dZ, RGrid):
    AR, Aphi, AZ = [vecField[:, :, i] for i in range(3)]
    solnR = scLapFD(AR, dR, dZ, RGrid) - AR / RGrid ** 2
    solnPhi = scLapFD(Aphi, dR, dZ, RGrid) - Aphi / RGrid ** 2
    solnZ = scLapFD(AZ, dR, dZ, RGrid)
    return np.stack((solnR, solnPhi, solnZ), axis = -1)

def gradVecFD(vecField, dR, dZ, RGrid):
    N = vecField.shape[0]
    ddR = FinDiff(1, dR)
    ddZ = FinDiff(0, dZ)
    dAdR = ddR(vecField)
    dAdZ = ddZ(vecField)
    soln = np.zeros([N, N, 3, 3])
    soln[..., :, 0] = dAdR
    soln[..., 0, 1] = -vecField[..., 1] / RGrid
    soln[..., 1, 1] = vecField[..., 0] / RGrid
    soln[..., :, 2] = dAdZ
    return soln

def vecDotGradVecFD(A, B, dR, dZ, RGrid):
    AR, Aphi, AZ = [A[..., i] for i in range(3)]
    BR, Bphi, BZ = [B[..., i] for i in range(3)]
    ddR = FinDiff(1, dR)
    ddZ = FinDiff(0, dZ)
    BR_R, Bphi_R, BZ_R = [ddR(B)[..., i] for i in range(3)]
    BR_Z, Bphi_Z, BZ_Z = [ddZ(B)[..., i] for i in range(3)]
    solnR = AR * BR_R + AZ * BR_Z - Aphi * Bphi / RGrid
    solnPhi = AR * Bphi_R + AZ * Bphi_Z + Aphi * BR / RGrid
    solnZ = AR * BZ_R + AZ * BZ_Z
    return np.stack((solnR, solnPhi, solnZ), axis = -1)

def dotProduct(AGrid, BGrid):
    A1, A2, A3 = AGrid[:, :, 0], AGrid[:, :, 1], AGrid[:, :, 2]
    B1, B2, B3 = BGrid[:, :, 0], BGrid[:, :, 1], BGrid[:, :, 2]
    return A1 * B1 + A2 * B2 + A3 * B3

def crossProduct(AGrid, BGrid):
    A1, A2, A3 = AGrid[:, :, 0], AGrid[:, :, 1], AGrid[:, :, 2]
    B1, B2, B3 = BGrid[:, :, 0], BGrid[:, :, 1], BGrid[:, :, 2]
    S1 = A2 * B3 - A3 * B2
    S2 = A3 * B1 - A1 * B3
    S3 = A1 * B2 - A2 * B1
    return np.stack((S1, S2, S3), axis = -1)

def crossMatGrid(vecField):
    N = len(vecField)

    vecCross = np.zeros([N, N, 3, 3])

    vecR, vecphi, vecZ = [vecField[:, :, i] for i in range(3)]

    vecCross[:, :, 0, 1] = -vecZ
    vecCross[:, :, 0, 2] = vecphi
    vecCross[:, :, 1, 0] = vecZ
    vecCross[:, :, 1, 2] = -vecR
    vecCross[:, :, 2, 0] = -vecphi
    vecCross[:, :, 2, 1] = vecR

    return vecCross

def testVecProducts(A, B):
    dot = DotOp(A)
    cross = CrossOp(A)
    assert_array_almost_equal(dotProduct(A, B), dot.apply(B), err_msg = "Dot products not equal")
    assert_array_almost_equal(crossProduct(A, B), cross.apply(B), err_msg = "Cross products not equal")
    print("Vector product tests passed")
    return

def testFirstOrderOps(A, B, dR, dZ, RGrid):
    N = A.shape[0]
    curl = CurlOp(N, dR, dZ, RGrid)
    div = DivOp(N, dR, dZ, RGrid)
    scGrad = ScGradOp(N, dR, dZ)
    vecGrad = VecGradOp(N, dR, dZ, RGrid)
    dirDer = DirDerivOp(N, dR, dZ, R, A)
    assert_array_almost_equal(curlFD(A, dR, dZ, RGrid), curl.apply(A), err_msg = "Curls not equal")
    assert_array_almost_equal(divFD(A, dR, dZ, RGrid), div.apply(A), err_msg = "Divergences not equal")
    assert_array_almost_equal(gradFD(A[..., 1], dR, dZ), scGrad.apply(A[..., 1]), err_msg = "Scalar gradients not equal")
    assert_array_almost_equal(gradVecFD(A, dR, dZ, RGrid), vecGrad.apply(A), err_msg = "Vector gradients not equal")
    assert_array_almost_equal(vecDotGradVecFD(A, B, dR, dZ, RGrid), dirDer.apply(B), err_msg = "Directional derivatives not equal")
    print("First order differential vector operator tests passed")
    return

def testSecondOrderOps(A, B, dR, dZ, RGrid):
    N = A.shape[0]
    scLap = ScalarLapOp(N, dR, dZ, RGrid)
    vecLap = VectorLapOp(N, dR, dZ, RGrid)
    gradDiv = GradDivOp(N, dR, dZ, RGrid)
    BMat = crossMatGrid(B)
    gradDivCoeff = GradDivOpWCoeff(N, dR, dZ, RGrid, A[..., 1], BMat)
    assert_array_almost_equal(scLapFD(A[..., 1], dR, dZ, RGrid), scLap.apply(A[..., 1]), err_msg = "Scalar Laplacians not equal")
    assert_array_almost_equal(vecLapFD(A, dR, dZ, RGrid), vecLap.apply(A), err_msg = "Vector Laplacians not equal")
    assert_array_almost_equal(gradFD(divFD(A, dR, dZ, RGrid), dR, dZ), gradDiv.apply(A), err_msg = "Gradients of divergence #1 not equal")
    assert_array_almost_equal(gradFD(A[..., 1] * divFD(crossProduct(B, A), dR, dZ, RGrid), dR, dZ), gradDivCoeff.apply(A), err_msg = "Gradients of divergence #2 not equal")
    print("Second order differential vector operator tests passed")
    return


if __name__ == '__main__':
    Rmin = 1
    Rmax = 2
    Zmin = 0
    Zmax = 1
    N = 64
    R = np.linspace(Rmin, Rmax, N)
    Z = np.linspace(Zmin, Zmax, N)
    RGrid, ZGrid = np.meshgrid(R, Z)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    A0 = np.stack((RGrid + ZGrid ** 2, np.cos(RGrid * ZGrid), ZGrid + 3 * RGrid ** 2), axis = -1)
    A1 = np.stack((np.sin(RGrid * ZGrid), 5 * RGrid ** 2 * ZGrid, RGrid ** 2 - ZGrid ** 2), axis = -1)
    testVecProducts(A0, A1)
    testFirstOrderOps(A1, A0, dR, dZ, RGrid)
    testSecondOrderOps(A1, A0, dR, dZ, RGrid)
