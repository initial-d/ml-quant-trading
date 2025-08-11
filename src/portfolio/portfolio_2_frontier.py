##
#  Copyright : Copyright (c) MOSEK ApS, Denmark. All rights reserved.
#
#  File :      portfolio_2_frontier.py
#
#  Purpose :   Implements a basic portfolio optimization model.
#              Computes points on the efficient frontier.
##
import mosek

def streamprinter(text):
    print("%s" % text),

if __name__ == '__main__':

    n = 3
    mu = [0.1073, 0.0737, 0.0627]
    GT = [[0.1667, 0.0232, 0.0013],
          [0.0000, 0.1033, -0.0022],
          [0.0000, 0.0000, 0.0338]]
    x0 = [0.0, 0.0, 0.0]
    w = 1.0
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

    inf = 0.0 # This value has no significance

    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, streamprinter)

            rtemp = w + sum(x0)

            # Constraints.
            task.appendcons(1 + n)
            task.putconbound(0, mosek.boundkey.fx, rtemp, rtemp)
            task.putconname(0, "budget")

            task.putconboundlist(range(1 + 0, 1 + n), n *
                                 [mosek.boundkey.fx], n * [0.0], n * [0.0])
            for j in range(1, 1 + n):
                task.putconname(j, "GT[%d]" % j)

            # Variables.
            task.appendvars(2 + 2 * n)

            offsetx = 0   # Offset of variable x into the API variable.
            offsets = n   # Offset of variable s into the API variable.
            offsett = n + 1 # Offset of variable t into the API variable.
            offsetu = 2*n + 1 # Offset of variable u into the API variable.

            # x variables.
            task.putclist(range(offsetx + 0, offsetx + n), mu)
            task.putaijlist(
                n * [0], range(offsetx + 0, offsetx + n), n * [1.0])
            for j in range(0, n):
                task.putaijlist(
                    n * [1 + j], range(offsetx + 0, offsetx + n), GT[j])

            task.putvarboundsliceconst(offsetx, offsetx + n, mosek.boundkey.lo, 0.0, inf)

            for j in range(0, n):
                task.putvarname(offsetx + j, "x[%d]" % (1 + j))

            # s variable.
            task.putvarbound(offsets + 0, mosek.boundkey.fr, -inf, inf)
            task.putvarname(offsets + 0, "s")

            # u variable.
            task.putvarbound(offsetu + 0, mosek.boundkey.fx, 0.5, 0.5)
            task.putvarname(offsetu + 0, "u")

            # t variables.
            task.putaijlist(range(1, n + 1), range(offsett +
                                                   0, offsett + n), n * [-1.0])
            task.putvarboundsliceconst(offsett, offsett + n, mosek.boundkey.fr, -inf, inf)
            for j in range(0, n):
                task.putvarname(offsett + j, "t[%d]" % (1 + j))

            task.appendcone(mosek.conetype.rquad, 0.0, 
                            [offsets, offsetu] + list(range(offsett, offsett + n)))
            task.putconename(0, "variance")

            task.putobjsense(mosek.objsense.maximize)

            # Turn all log output off.
            task.putintparam(mosek.iparam.log, 0)

            for alpha in alphas:
                # Dump the problem to a human readable OPF file.
                #task.writedata("dump.opf")

                task.putcj(offsets + 0, -alpha)

                task.optimize()

                # Display the solution summary for quick inspection of results.
                # task.solutionsummary(mosek.streamtype.msg)

                solsta = task.getsolsta(mosek.soltype.itr)

                if solsta in [mosek.solsta.optimal]:
                    expret = 0.0
                    x = [0.] * n
                    task.getxxslice(mosek.soltype.itr,
                                    offsetx + 0, offsetx + n, x)
                    for j in range(0, n):
                        expret += mu[j] * x[j]

                    stddev = [0.]
                    task.getxxslice(mosek.soltype.itr,
                                    offsets + 0, offsets + 1, stddev)

                    print("alpha = {0:.2e} exp. ret. = {1:.3e}, variance {2:.3e}".format(alpha, expret, stddev[0]))
                else:
                    print("An error occurred when solving for alpha=%e\n" % alpha)
