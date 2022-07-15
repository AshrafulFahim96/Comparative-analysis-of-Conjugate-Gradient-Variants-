import numpy as np
from scipy import sparse
from CG_Module import conjGrad


def shrink(x, r):
    z = np.multiply(np.sign(x), np.maximum(np.abs(x) - r, 0))
    return z


def defDDt(N):
    # Create a first order difference matrix D (Nabla)
    e = np.ones(N)
    B = sparse.spdiags(np.array([e, - e]), np.array([1, 0]), N, N)
    B = sparse.lil_matrix(B)
    B[N-1, 1] = 1   # Here it is from 0-1023. In MatLab it is 1-1024

    D = B
    DT = np.transpose(D)  # Create the transpose of D
    DTD = np.dot(np.transpose(D), D)

    return D, DT, DTD


# Default method = FR
def AM_1D(b, A, mu, rho, Nit, tol, method="FR"):
    # Initializations
    x = b
    N = len(x)
    funcVal = np.zeros((Nit, 1))
    relError = np.zeros((Nit, 1))

    # The T stands for the transpose.
    # D = nabla, DT = nabla transpose, DTD = nabla transpose nabla.
    D, DT, DTD = defDDt(N)

    AT = np.transpose(A)
    ATA = np.dot(np.transpose(A), A)
    print("Iteration Started for: ", method)
    for k in np.arange(0, Nit).reshape(-1):

        # ##### z-sub-problem ######
        Dx = D * x
        z = shrink(Dx, 1 / rho)  # the "shrinkage" or "soft thresholding" operation.

        # ##### x-sub-problem ######
        # Here is where we solve the linear system.
        lhs = ATA + np.dot((rho / mu), DTD)
        rhs = np.dot(AT, b) + (1 / mu) * DT * z
        x_old = x

        # Here we solve the linear system using the conjugate gradient method.
        x = conjGrad(lhs, rhs, 1e-05, 10, meth=method)

        # Norm configured for Frobenius Norms
        relError[k] = np.linalg.norm(x - x_old) / np.linalg.norm(x)

        # Norm configured for 2-Norm
        funcVal[k] = (mu / 2) * np.linalg.norm(A * x - b, 2) ** 2 + \
                     (rho / 2) * np.linalg.norm(z - D * x, 2) ** 2 + \
                     sum(np.abs(D * x))

        if relError[k] < tol:
            break

    out = {
        "sol": x,
        "funVal": funcVal[np.arange(1, k + 1)],
        "relativeError": relError[np.arange(1, k + 1)]
    }

    return out
