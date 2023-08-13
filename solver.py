"""
**********  Solver  **********
"""

import numpy as np

def Conjugate_Gradient(A, b, x, tol=1e-8, priori_check=True):
    """
    conjugate gradient method

    args:
        A: coef
        b: src
        x: unknown

    returns:
        x: solution
    """

    def is_sym(M):
        """
        check if the given matrix M is symmetric

        args:
            M: matrix (2D array)
        """
        return np.all(M == M.T)

    def is_pos_def(M):
        """
        check if the given matrix M is positive definite
        checked via its eigenvalues (eigenvalues(M) > 0 => M is pos def)

        args:
            M: matrix (2D array)
        """
        return np.all(np.linalg.eigvals(M) > 0)

    print("\n>>>>> Conjugate Gradient method;")

    if priori_check:
        sym = is_sym(A)
        pos_def = is_pos_def(A)
        if sym and pos_def:
            print(">>>>> given matrix A is symmetric & positive definite, continue")
        else:
            print(">>>>> given matrix A is not symmetric or not positive definite, but continue anyway")
            # raise ValueError(">>>>> given A did NOT pass priori_check, abort")

    n = len(x)
    r0 = b - np.dot(A, x)
    p = r0

    log = []
    for it in range(0, 2*n):
        alpha = np.dot(r0, r0) / np.dot(p, (np.dot(A, p)))
        x += alpha * p
        r1 = r0 - alpha * np.dot(A, p)

        beta = np.dot(r1, r1) / np.dot(r0, r0)
        p = r1 + beta * p
        r0 = r1

        res = np.linalg.norm(r1, ord=2)
        log.append(res)
        if it % 10 == 0:
            print(f">>>>> it: {it}, res: {res:.6e}")
        if res < tol:
            print(">>>>> converged, loop terminating now")
            break

    return x, log
