"""
test solvers for linear systems of equations
"""

import argparse
import numpy as np
import scipy as sp

from solvers import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--symmetric",
        action="store_true",
        help="generate symmetric matrix"
    )
    args = parser.parse_args()
    return args

def main(args):
    if args.symmetric:
        print(">>>>> generate symmetric matrix")
        # generate random symmetric matrix
        n = 10
        A = np.random.rand(n, n)
        A = np.dot(A, A.T)
    else:
        print(">>>>> generate non-symmetric matrix")
        # generate random matrix (not necessarily symmetric)
        n = 10
        A = np.random.rand(n, n)

    # generate random source vector
    b = np.random.rand(n)

    # generate random initial guess
    x = np.zeros(n)

    # test CG
    print(">>>>> solve with CG")
    _x = CG(A, b, x, priori_check=False, tol=1e-8)
    print(_x)

    # solve with scipy.sparse.linalg.cg
    _x, code = sp.sparse.linalg.cg(A, b, x0=x, tol=1e-8)
    print(_x)

if __name__ == "__main__":
    args = parse_args()
    main(args)

