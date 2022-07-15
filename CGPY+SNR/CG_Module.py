import numpy as np


# Default method = FR
def conjGrad(A, b, tol, Nit, meth="FR"):
    # Initialization.
    x = np.random.random_sample(b.shape)
    r = b - np.dot(A, x)
    d = -r

    d_old = 0.0
    r_old = 0.0
    rho_old = 0.0

    for k in np.arange(0, Nit).reshape(-1):
        rho = np.dot(np.transpose(r), r)
        rho2 = np.dot(np.transpose(r), r - r_old)
        rho3 = np.dot(np.transpose(d_old), r - r_old)

        if k == 0:
            p = r
        else:
            # Check Method defined and calculate beta accordingly

            if meth == "FR":
                beta = rho / rho_old
            elif meth == "PPR":
                beta = rho2 / rho_old
            elif meth == "HS":
                beta = rho2 / rho3
                d = -r + (d_old * beta)
            elif meth == "DY":
                beta = rho / rho3
                d = -r + (d_old * beta)
            else:
                print("Unknown Method: Calculating Beta for FR by Default")
                beta = rho / rho_old

            p = r + (p * beta)

        q = np.dot(A, p)
        gamma = np.dot(np.transpose(p), q)
        alpha = rho / gamma

        x = x + (p * alpha)
        r_old = r
        r = r - (q * alpha)
        rho_old = rho
        d_old = d

        if r.any() < tol:
            break
    
    return x
