#!opt/conda/envs/project/bin/python
# -*- coding: utf-8 -*-
"""Stripe pattern"""

from functools import total_ordering
import numpy as np
from fenics import *
from mpi4py import MPI
comm = MPI.COMM_WORLD

class MeanFlow(UserExpression):
    def eval(self, val, x):
        val[0] = 1.0
        val[1] = 0.0
    def value_shape(self):
        return (2,)

class InitialConditions(UserExpression):
    def eval(self, val, x):
        val[0] = np.sqrt(1)*np.random.randn()
        val[1] = 0.0
    def value_shape(self):
        return (2,)

class SwiftHohenbergEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

class KrylovSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, comm, PETScKrylovSolver(), PETScFactory.instance())
    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)
        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("pc_type", "ilu")
        self.linear_solver().set_from_options()

class Terms():
    def __init__(self, epsilon, weight, direction, frequency, g=0.0):
        self.q = frequency
        self.direction = direction
        self.eps = epsilon
        self.weight = weight
        self.g = g

    def term_A(self, w, v):
        return (dot(grad(w), grad(v)))*dx - ((self.q**2)*w*v)*dx

    def term_A_aniso(self, w, v):
        u = self.direction
        D = outer(u, u)
        return 2*self.weight**2*(dot(grad(w), dot(D, grad(v))))*dx
    
    def Phi(self, w):
        return -self.eps*w**2/2 + w**4/4 - self.g/3*w**3

    def Phi_p(self, w):
        return -self.eps*w + w**3 -self.g * w**2

    def G1(self, w, v):
        return -self.eps/2 + (w**2 + w*v + v**2)/4 - self.g/3 * (w + v)

    def G2(self, v):
        return -self.eps*v/2 + v**3/4 - self.g/3 * v**2

def swift_hohenberg():
    '''compiler options'''
    # We use the Newton-Krylov method
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters['allow_extrapolation'] = True
    
    solver = KrylovSolver()

    mesh = Mesh('/workdir/meshs/implant_step.xml')
    orientation_path = '/workdir/results/implant/100/orient.xml'
    density_path = '/workdir/results/implant/100/dens.xml'
    output_path = '/workdir/results/implant/100'

    weight = 1.6
    elapsed = 0.0
    total_time = 1
    dt = 0.001
    width_0 = 0.8

    V = FiniteElement('CG', mesh.ufl_cell(), 2)
    ME = FunctionSpace(mesh, V*V)
    Refine_V = FunctionSpace(refine(mesh), 'CG', 1)
    Orientation_Space = VectorFunctionSpace(mesh, 'CG', 1)
    Density_space = FunctionSpace(mesh, 'CG', 1)

    #density = Function(Density_space, density_path)
    density = Constant(1.0)
    orientation = Function(Orientation_Space, orientation_path)
    orientation_inverse = project(as_vector([orientation[1], -orientation[0]]), Orientation_Space)
    #orientation = Function(Orientation_Space)
    #flow = MeanFlow()
    #orientation.interpolate(flow)
    frequency = sqrt((density/width_0)**2*np.pi**2 - weight**2)

    U = TrialFunction(ME)
    phi, psi = TestFunctions(ME)
    Uh = Function(ME)
    Uh_0 = Function(ME)
    uh, qh = split(Uh)
    uh_0, qh_0 = split(Uh_0)
    U_init = InitialConditions()
    Uh.interpolate(U_init)
    Uh_0.interpolate(U_init)    

    file = XDMFFile(output_path + '/field.xdmf')
    lyapnov_path = output_path + '/lyapnpv.csv'
    lyapnov_list = []

    terms = Terms(10, weight, orientation_inverse, frequency, g=0.0)

    verbosity = range(0, int(total_time//dt), 5)
    index = 0
    while (elapsed < total_time):
        qh_mid = 0.5*qh + 0.5*qh_0
        dPhi = terms.G1(uh, uh_0)*uh + terms.G2(uh_0)
        F1 = (uh-uh_0)*phi*dx + dt*terms.term_A(qh_mid, phi) + dt*dPhi*phi*dx - dt*terms.term_A_aniso(phi, uh)
        F2 = qh*psi*dx - terms.term_A(uh, psi)    
        F = F1 + F2
        a = derivative(F, Uh, U)
        problem = SwiftHohenbergEquation(a, F)
        elapsed += dt
        solver.solve(problem, Uh.vector())
        uh_solved = Uh.split()[0]
        k = frequency
        L = (0.5*(div(grad(uh_solved))**2 - 2*k**2*(dot(grad(uh_solved), grad(uh_solved)))\
             + k**4*uh_solved**2) + terms.Phi(uh_solved) - weight**2*dot(grad(uh_solved), dot(outer(orientation, orientation), grad(uh_solved))))*dx
        lyapnov_list.append(assemble(L))
        Uh_0.vector()[:] = Uh.vector()
        if index in list(verbosity):
            refine_uh_solved = interpolate(uh_solved, Refine_V)
            refine_uh_solved.rename('OrderParameter', 'label')
            file.write(refine_uh_solved, elapsed)
        index += 1
        np.savetxt(lyapnov_path, lyapnov_list, delimiter=',')
    pass

if __name__ == '__main__':
    swift_hohenberg()