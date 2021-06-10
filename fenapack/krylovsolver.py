"""
BSD 3-Clause License

Copyright (c) 2020, Fabio Durastante, Stefano Cipolla
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
# Krylov solver for nonlinear problems
from dolfin import *
from scipy.sparse import csr_matrix
from scipy.optimize import least_squares
from pynonlinkrylov import krylovsubspace
from pyamg import ruge_stuben_solver
import numpy as np


def ConvergenceFunction(iteration, v, v0, abs_tol, rel_tol):
    """
    Evaluate convergence for a given non-linear method
    :param iteration: Number of the iteration
    :param v: Actual iteration
    :param v0: Initial guess, on iteration 0 computes the norm, on the other propagates it
    :param abs_tol: Absolute tolerance
    :param rel_tol: Relative tolerance
    :return convergence: True if absolute < abs_tol or relative < rel_tol
    :return absolute: Absolute Error
    :return relative: Relative Error
    """
    absolute = v.norm("l2")
    if iteration == 0:
        absolute0 = absolute
    else:
        absolute0 = v0
    relative = absolute / absolute0
    if absolute < abs_tol or relative < rel_tol:
        convergence = True
    else:
        convergence = False
    return convergence, absolute, relative


class NonLinearKrylovInfo:
    """
    Thi class contains the information relative to one execution
    of any the method implemented here. Is returned by the call
    to the non-linear method and posses a print method that outs
    the information to screen.
    """
    krylov_space_size = []
    number_of_function_evaluation = []
    absolute_error = []
    relative_error = []
    total_number_of_function_eval = 0
    number_of_iteration = 0

    def print(self, *argv: str):
        """
        This function prints on screen the information relative to
        the execution of the algorithm as stored in the class.
        :parameter argv: expects a string with filename to print
                         the results on file.
        :return: prints on screen
        """
        if len(argv) > 2:
            raise Exception('Function called with > 2 inputs')

        print("Size of the Krylov spaces: ", self.krylov_space_size)
        print("Number of function evaluations per iteration: ", self.number_of_function_evaluation)
        print("Total Number of function evaluations: ", self.total_number_of_function_eval)

    def append(self, which: str, what):
        """
        This function appends to class variable "which" the value "what"
        :param which: Name of the class variable: "Krylov", "Feval", "Absolute", "Relative"
        :param what: the value to append
        :return: the updated class
        """
        if which.upper() == "KRYLOV":
            self.krylov_space_size.append(what)
        elif which.upper() == "FEVAL":
            self.number_of_function_evaluation.append(what)
            self.total_number_of_function_eval += what
        elif which.upper() == "ABSOLUTE":
            self.absolute_error.append(what)
        elif which.upper() == "RELATIVE":
            self.relative_error.append(what)
        else:
            raise Exception('Unrecognized value')

    def clear(self):
        """
        Clear the variables in the info structure
        :return: self
        """
        self.krylov_space_size = []
        self.number_of_function_evaluation = []
        self.absolute_error = []
        self.relative_error = []
        self.total_number_of_function_eval = 0
        self.number_of_iteration = 0


def NonLinearKrylov(ffunc, jfunc, V, bc, bc_du, criterion, iteration_max=10,
                    absolute_tol=1.0E-10, relative_tol=1.0E-6, krylov_tol=1.0E-3,
                    optim_tol=1.0E-3):
    """
    NonLinearKrylov Solver with polynomial Krylov space as search space
    :parameter ffunc: Bilinear form F(u,v)=0
    :parameter jfunc: Jacobian wrt u of the bilinear form ffunc
    :parameter V: Trial function space
    :parameter bc: Boundary conditions for the problem
    :parameter bc_du: Boundaary conditions for the residual
    :parameter criterion: Values are "residual" or "incremental", see ConvergenceFunction
    :parameter iteration_max: Maximum number of iterations
    :parameter absolute_tol: Absolute tolerance on F(u,v)=0
    :parameter relative_tol: Relative tolerance on F(u,v)=0
    :parameter krylov_tol: Tolerance on Krylov subspace generation
    :parameter optim_tol: Tolerance of LineSearch method
    :return u: containing the solution to the non-linear variational problem
    :return info: class object NonLinearKrylov containing the information
    """
    info = NonLinearKrylovInfo()
    info.clear()

    v = TestFunction(V)
    du = TrialFunction(V)
    du_ = Function(V)
    u_ = Function(V)

    u = interpolate(Constant(0.0), V)
    bc.apply(u.vector())
    iteration = 0
    absolute = 1.0
    convergence = False
    info.append("absolute", assemble(ffunc(u, v)).norm("l2"))
    while iteration < iteration_max and convergence != True:
        F = ffunc(u, v)
        J = jfunc(F, u, du)
        A, b = assemble_system(J, -F, bc_du)
        # Convert vector and matrix to sparse memorization
        A_mat = as_backend_type(A).mat()
        A_sparray = csr_matrix(A_mat.getValuesCSR()[::-1], shape=A_mat.size)
        b_array = b.get_local()
        # Compute the Krylov subspace as search space
        Q, h, k = krylovsubspace.arnoldi_iteration(A_sparray, b_array, krylov_tol, 100)
        # Find the search direction solving the auxiliary optimization problem
        fx_norm = norm(b, "l2")
        info.append("krylov", k)

        def objective(x):
            y = Function(V)
            y.vector()[:] = u.vector()[:] + Q.dot(x)
            dy = assemble(ffunc(y, v))
            bc_du.apply(dy)  # The increment has to satisfy 0 Dirichlet BCs to avoid adding boundary terms
            return dy.get_local()
        # We avoid a request on the convergence for the subspace search going to zero (machine tolerance)
        ftol = np.max([optim_tol * fx_norm, absolute_tol])
        optimization_result = least_squares(objective, np.zeros(k), method='lm', ftol=ftol)
        du_.vector()[:] = Q.dot(optimization_result['x'])
        u_.vector()[:] = u.vector()[:] + du_.vector()[:]
        u.assign(u_)
        if criterion == "residual":
            b = assemble(-ffunc(u, v))
            bc_du.apply(b)
            convergence, absolute, relative = ConvergenceFunction(iteration, b, absolute, absolute_tol,
                                                                  relative_tol)
        elif criterion == "incremental":
            convergence, absolute, relative = ConvergenceFunction(iteration, du_.vector(), absolute, absolute_tol,
                                                                  relative_tol)
        else:
            print("Convergence criterion not valid")
            sys.exit()
        print("Newton iteration %d: r (abs) = %.3e (tol = %.3e) r (rel) = %.3e (tol = %.3e) Krylov = %d Feval = %d" % (
            iteration, absolute, absolute_tol, relative, relative_tol, k, optimization_result['nfev'] + 1))
        # Append information on the iteration to the data structure
        info.append("feval", optimization_result['nfev'] + 1)
        info.append("absolute", absolute)
        info.append("relative", relative)
        iteration += 1
    info.number_of_iteration = iteration
    return u, info