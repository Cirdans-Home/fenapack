# -*- coding: utf-8 -*-

# Copyright (C) 2014-2016 Jan Blechta and Martin Řehoř
#
# This file is part of FENaPack.
#
# FENaPack is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FENaPack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FENaPack.  If not, see <http://www.gnu.org/licenses/>.

"""This is FENaPack, FEniCS Navier-Stokes preconditioning package."""

__author__ = "Jan Blechta, Martin Řehoř"
__version__ = "2017.1.0.dev0"
__license__ = "GNU LGPL v3"

# Do not use petsc4py python error handler (hides error messages
# to PETSc C calls), workaround to DOLFIN issue #801
from dolfin import SubSystemsManager
SubSystemsManager.init_petsc()  # init PETSc by DOLFIN before petsc4py import
from petsc4py import PETSc
PETSc.Sys.pushErrorHandler("traceback")
del SubSystemsManager, PETSc

# Import public API
from fenapack.field_split import PCDKSP, PCDKrylovSolver
from fenapack.nonlinear_solvers import PCDNewtonSolver, PCDProblem
from fenapack.preconditioners import PCDPC_BRM1, PCDPC_BRM2
from fenapack.stabilization import StabilizationParameterSD
