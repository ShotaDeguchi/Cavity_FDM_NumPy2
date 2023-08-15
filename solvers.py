"""
solvers for linear systems of equations
"""

import numpy as np

def CG(A, b, x, tol=1e-8, priori_check=True):
    """
    Conjugate Gradient method

    args:
        A: coefficient / adjacency
        b: source
        x: solution (initial guess)

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

    # CG is proven to converge within finite number of iterations (len of x vector)
    for it in range(0, n):
        alpha = np.dot(r0, r0) / np.dot(p, (np.dot(A, p)))
        x += alpha * p
        r1 = r0 - alpha * np.dot(A, p)

        beta = np.dot(r1, r1) / np.dot(r0, r0)
        p = r1 + beta * p
        r0 = r1

        res = np.linalg.norm(r1, ord=2)
        if it % 10 == 0:
            print(f">>>>> CG method it: {it}, res: {res:.6e}")
        if res < tol:
            print(f">>>>> CG method converged")
            break

    return x

################################################################################

def BiCGSTAB(A, b, x, tol=1e-8):
    """
    Biconjugate Gradient Stabilized method
    Does not require A to be symmetric

    args:
        A: coefficient / adjacency
        b: source
        x: solution (initial guess)

    returns:
        x: solution
    """

    print("\n>>>>> BiCGSTAB method;")

    n = len(x)
    r0 = b - np.dot(A, x)
    r0_hat = r0
    rho0 = 1
    alpha = 1
    omega = 1
    v0 = np.zeros(n)
    p0 = np.zeros(n)
    for it in range(0, n):
        rho = np.dot(r0_hat, r0)
        beta = (rho / rho0) * (alpha / omega)
        p = r0 + beta * (p0 - omega * v0)
        v = np.dot(A, p)
        alpha = rho / np.dot(r0_hat, v)
        s = r0 - alpha * v
        t = np.dot(A, s)
        omega = np.dot(t, s) / np.dot(t, t)
        x += alpha * p + omega * s
        r0 = s - omega * t
        rho0 = rho
        p0 = p
        v0 = v

        res = np.linalg.norm(r0, ord=2)
        if it % 10 == 0:
            print(f">>>>> BiCGSTAB method it: {it}, res: {res:.6e}")
        if res < tol:
            print(f">>>>> BiCGSTAB method converged")
            break

    return x

################################################################################

def Jacobi(A, b, x, tol=1e-8, maxiter=1000):
    """
    Jacobi method

    args:
        A: coefficient / adjacency
        b: source
        x: solution (initial guess)

    returns:
        x: solution
    """

    # A = D + (L + U) = D + E
    D = np.diag(A)         # diag elements
    D = np.diagflat(D)     # diag matrix
    E = A - D              # E = L + U
    D_inv = np.linalg.inv(D)

    print("\n>>>>> Jacobi method;")
    for it in range(0, maxiter):
        x_old = np.copy(x)
        x = np.matmul(D_inv, (b - np.matmul(E, x_old)))
        res = np.sqrt(np.sum((x - x_old)**2)) / np.sqrt(np.sum(x_old**2))

        if it % int(maxiter / 10) == 0:
            print(f">>>>> Jacobi method it: {it}, res: {res:.6e}")
            if res < tol:
                print(f">>>>> Jacobi method converged")
                break

    return x



def GaussSeidel(A, b, x, tol=1e-8, maxiter=1000):
    """
    Gauss-Seidel method

    args:
        A: coefficient / adjacency
        b: source
        x: solution (initial guess)

    returns:
        x: solution
    """

    D = np.diag(A)         # diag elements
    D = np.diagflat(D)     # diag matrix
    L = np.tril(A, k=-1)   # strictly lower triangular
    U = np.triu(A, k=1)    # strictly upper triangular
    P = L + D
    Q = np.linalg.inv(P)

    print("\n>>>>> Gauss-Seidel method;")

    for it in range(0, maxiter):
        x_old = np.copy(x)
        x = np.matmul(Q, (b - np.matmul(U, x_old)))
        res = np.sqrt(np.sum((x - x_old)**2)) / np.sqrt(np.sum(x_old**2))

        if it % int(maxiter / 10) == 0:
            print(f">>>>> Gauss-Seidel method it: {it}, res: {res:.6e}")
            if res < tol:
                print(f">>>>> Gauss-Seidel method converged")
                break
