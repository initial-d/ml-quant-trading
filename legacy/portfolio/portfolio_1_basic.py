##
#  Copyright : Copyright (c) MOSEK ApS, Denmark. All rights reserved.
#
#  File :      portfolio_1_basic.py
#
#  Purpose :   Implements a basic portfolio optimization model.
##
import mosek
import sys

def streamprinter(text):
    sys.stdout.write("%s" % text),

if __name__ == '__main__':

    n = 3
    gamma = 0.05
    mu = [0.1073, 0.0737, 0.0627]
    GT = [[0.1667, 0.0232, 0.0013],
          [0.0000, 0.1033, -0.0022],
          [0.0000, 0.0000, 0.0338]]
    x0 = [0.0, 0.0, 0.0]
    w = 1.0

    inf = 0.0 # This value has no significance

    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, streamprinter)

            # Constraints.
            task.appendcons(1 + n)

            # Total budget constraint - set bounds l^c = u^c
            rtemp = w + sum(x0)
            task.putconbound(0, mosek.boundkey.fx, rtemp, rtemp)
            task.putconname(0, "budget")

            # The remaining constraints GT * x - t = 0 - set bounds l^c = u^c
            task.putconboundlist(range(1 + 0, 1 + n), [mosek.boundkey.fx] * n, [0.0] * n, [0.0] * n)
            for j in range(1, 1 + n):
                task.putconname(j, "GT[%d]" % j)

            # Variables.
            task.appendvars(1 + 2 * n)

            # Offset of variables into the API variable.
            offsetx = 0
            offsets = n
            offsett = n + 1

            # x variables.
            # Returns of assets in the objective 
            task.putclist(range(offsetx + 0, offsetx + n), mu)
            # Coefficients in the first row of A
            task.putaijlist([0] * n, range(offsetx + 0, offsetx + n), [1.0] * n)
            # No short-selling - x^l = 0, x^u = inf 
            task.putvarboundslice(offsetx, offsetx + n, [mosek.boundkey.lo] * n, [0.0] * n, [inf] * n)
            for j in range(0, n):
                task.putvarname(offsetx + j, "x[%d]" % (1 + j))

            # s variable is a constant equal to gamma
            task.putvarbound(offsets + 0, mosek.boundkey.fx, gamma, gamma)
            task.putvarname(offsets + 0, "s")

            # t variables (t = GT*x).
            # Copying the GT matrix in the appropriate block of A
            for j in range(0, n):
                task.putaijlist(
                    [1 + j] * n, range(offsetx + 0, offsetx + n), GT[j])
            # Diagonal -1 entries in a block of A
            task.putaijlist(range(1, n + 1), range(offsett + 0, offsett + n), [-1.0] * n)
            # Free - no bounds
            task.putvarboundslice(offsett + 0, offsett + n, [mosek.boundkey.fr] * n, [-inf] * n, [inf] * n)
            for j in range(0, n):
                task.putvarname(offsett + j, "t[%d]" % (1 + j))

            # Define the cone spanned by variables (s, t), i.e. dimension = n + 1 
            task.appendcone(mosek.conetype.quad, 0.0, [offsets] + list(range(offsett, offsett + n)))
            task.putconename(0, "stddev")

            task.putobjsense(mosek.objsense.maximize)

            # Dump the problem to a human readable OPF file.
            task.writedata("dump.opf")

            task.optimize()

            # Display solution summary for quick inspection of results.
            task.solutionsummary(mosek.streamtype.msg)

            # Retrieve results
            xx = [0.] * (n + 1)
            task.getxxslice(mosek.soltype.itr, offsetx + 0, offsets + 1, xx)
            expret = sum(mu[j] * xx[j] for j in range(offsetx, offsetx + n))
            stddev = xx[offsets]

            print("\nExpected return %e for gamma %e\n" % (expret, stddev))
    sys.stdout.flush()
