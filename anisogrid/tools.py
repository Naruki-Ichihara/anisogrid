#!opt/conda/envs/project/bin/python
# -*- coding: utf-8 -*-
"""Topology and material orientation optimization tool set."""

from fenics import *
from fenics_adjoint import *
import numpy as np
from ufl import operators, transpose
from torch_fenics import *
import torch
import nlopt as nl

def iso_filter(z, e):
    """# iso_filter

    Apply 2D isoparametric projection onto orientation vector.

    Args:
        z: 0-component of the orientation vector (on natural setting)
        e: 1-component of the orientation vector (on natural setting)

    Returns:
        [Nx, Ny] (fenics.vector): Orientation vector with unit circle boundary condition on real setting.
    """
    u = as_vector([-1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 1/np.sqrt(2), 0, -1/np.sqrt(2), -1])
    v = as_vector([-1/np.sqrt(2), -1, -1/np.sqrt(2), 0, 1/np.sqrt(2), 1, 1/np.sqrt(2), 0])
    N1 = -(1-z)*(1-e)*(1+z+e)/4
    N2 =  (1-z**2)*(1-e)/2
    N3 = -(1+z)*(1-e)*(1-z+e)/4
    N4 =  (1+z)*(1-e**2)/2
    N5 = -(1+z)*(1+e)*(1-z-e)/4
    N6 =  (1-z**2)*(1+e)/2
    N7 = -(1-z)*(1+e)*(1+z-e)/4
    N8 =  (1-z)*(1-e**2)/2
    N = as_vector([N1, N2, N3, N4, N5, N6, N7, N8])
    Nx = inner(u, N)
    Ny = inner(v, N)
    return as_vector((Nx, Ny))

def helmholtz_filter(u, U, r=0.025):
    """# helmholtz_filter

    Apply the helmholtz filter.

    Args:
        u (fenics.function): target function
        U (fenics.FunctionSpace): Functionspace of target function
        r (float): filter radius

    Return:
        v (fenics.function): filtered function
    """
    v = TrialFunction(U)
    dv = TestFunction(U)
    vh = Function(U)
    a = r*inner(grad(v), grad(dv))*dx + dot(v, dv)*dx
    L = dot(u, dv)*dx
    solve(a==L, vh)
    return project(vh, U)

def heviside_filter(u, U, offset=0.01):
    """# heviside_filter
    
    Apply the heviside function

    Args:
        u (fenics.function): target function
        U (fenics.FunctionSpace): Functionspace of target function
        offset (float): base line

    Returns:
        v (fenics.function): filterd function

    Note:
        TODO: Using Regularlized Heaviside function
    
    """
    return project((1-offset)*u/2 + (1+offset)/2, U)

def rotated_lamina_stiffness_inplane(E1, E2, G12, nu12, theta):
    """# rotated_lamina_stiffness_inplane
    
    Return the in-plane stiffness matrix of an orhtropic layer
    in a reference rotated by an angle theta wrt to the material one.
    It assumes Voigt notation and plane stress state.
    (see Reddy 1997, eqn 1.3.71)

    Args:
        E1: The Young modulus in the material direction 1.
        E2: The Young modulus in the material direction 2.
        G23: The in-plane shear modulus.
        nu12: The in-plane Poisson ratio.
        theta: The rotation angle from the material to the desired reference system.

    Returns:
        Q_theta: a 3x3 symmetric ufl matrix giving the stiffness matrix.
    """
    c = cos(theta)
    s = sin(theta)
    T = as_matrix([[c**2, s**2, -2*s*c],
                   [s**2, c**2, 2*s*c],
                   [c*s, -c*s, c**2 - s**2]])
    # In-plane stiffness matrix of an orhtropic layer in material coordinates
    nu21 = E2/E1*nu12
    Q11 = E1/(1-nu12*nu21)
    Q12 = nu12*E2/(1-nu12*nu21)
    Q22 = E2/(1-nu12*nu21)
    Q66 = G12
    Q16 = 0.
    Q26= 0.
    Q = as_matrix([[Q11, Q12, Q16],
                   [Q12, Q22, Q26],
                   [Q16, Q26, Q66]])
    # Rotated matrix in the main reference
    Q_theta = T*Q*transpose(T)

    return Q_theta

def strain_to_voigt(e):
    """# strain_to_voigt
    
    Returns the pseudo-vector in the Voigt notation associate to a 2x2
    symmetric strain tensor, according to the following rule (see e.g.https://en.wikipedia.org/wiki/Voigt_notation),

    Args:
        e: a symmetric 2x2 strain tensor, typically UFL form with shape (2,2)

    Returns:
        a UFL form with shape (3,1) corresponding to the input tensor in Voigt
        notation.
    """
    return as_vector((e[0,0], e[1,1], 2*e[0,1]))


def stress_to_voigt(sigma):
    """# stress_to_voigt
    
    Returns the pseudo-vector in the Voigt notation associate to a 2x2
    symmetric stress tensor, according to the following rule (see e.g.https://en.wikipedia.org/wiki/Voigt_notation),

    Args:
        sigma: a symmetric 2x2 stress tensor, typically UFL form with shape
        (2,2).

    Returns:
        a UFL form with shape (3,1) corresponding to the input tensor in Voigt notation.
    """
    return as_vector((sigma[0,0], sigma[1,1], sigma[0,1]))


def strain_from_voigt(e_voigt):
    """# strain_from_voigt
    
    Inverse operation of strain_to_voigt.

    Args:
        sigma_voigt: UFL form with shape (3,1) corresponding to the strain
        pseudo-vector in Voigt format

    Returns:
        a symmetric stress tensor, typically UFL form with shape (2,2)
    """
    return as_matrix(((e_voigt[0], e_voigt[2]/2.), (e_voigt[2]/2., e_voigt[1])))


def stress_from_voigt(sigma_voigt):
    """# stress_from_voigt
    
    Inverse operation of stress_to_voigt.

    Args:
        sigma_voigt: UFL form with shape (3,1) corresponding to the stress
        pseudo-vector in Voigt format.

    Returns:
        a symmetric stress tensor, typically UFL form with shape (2,2)
    """
    return as_matrix(((sigma_voigt[0], sigma_voigt[2]), (sigma_voigt[2], sigma_voigt[1])))

def strain(u):
    """# strain
    
    Calculate strain tensor from inplane displacement vector.

    Args:
        u: inplane displacement vector fenics function

    Returns:
        eps: inplane strain fenics tensor function
    """
    return sym(grad(u))

def stress(Q, u):
    """# stress
    
    Calculate stress tensor from inplane displacement tensor and inplane stiffness matrix

    Args:
        Q: 3*3 fenics materix for inplane stiffness
        u: inplane displacement vector fenics function

    returns:
        sigma: inplane stress fenics tensor function
    """
    return stress_from_voigt(Q*strain_to_voigt(sym(grad(u))))


class CoreProcess(torch_fenics.FEniCSModule):
    """# Core class
    
    Apply some pre-process and eveluate the loss function.

    Attributes:
        mesh (fenics.Mesh): mesh for FE analysis.
        load_conditions (list[fenics.bc]):
        applied_load_vector (list[fenics.vector]):
        bcs (list[fenics.bc]): boundary condtions.
        material_parameters (dict{'float}): material_parameters.
        files (dict{'str'}): File paths for saving some results.
    
    """
    def __init__(self, mesh, load_conditions, applied_load_vectors, displacement_boundaries_sub0, displacement_boundaries_sub1, applied_displacements_sub0, applied_displacements_sub1, material_parameters, files):
        super().__init__()
        self.mesh = mesh
        self.load_conditions = load_conditions
        self.applied_loadvectors = applied_load_vectors
        self.displacement_boundaries_sub0 = displacement_boundaries_sub0
        self.applied_displacements_sub0 = applied_displacements_sub0
        self.displacement_boundaries_sub1 = displacement_boundaries_sub1
        self.applied_displacements_sub1 = applied_displacements_sub1
        self.material_parameters = material_parameters
        self.files = files
    
    def input_templates(self):
        return Function(FunctionSpace(self.mesh, 'CG', 1)), Function(FunctionSpace(self.mesh, 'CG', 1)), Function(FunctionSpace(self.mesh, 'CG', 1))

    def solve(self, z, e, r):
        """# solve

        calcuate strain energy from given design parameters.

        Args:
            z: 0-component of the orientation vector (on natural setting)
            e: 1-component of the orientation vector (on natural setting)
            r: Relatively density field

        Returns:
            f: Strain energy
        """

        facets = MeshFunction('size_t', self.mesh, 1)
        facets.set_all(0)
        for i in range(len(self.load_conditions)):
            self.load_conditions[i].mark(facets, int(i+1))
        ds = Measure('ds', subdomain_data=facets)

        Orient = VectorFunctionSpace(self.mesh, 'CG', 1)
        orient = helmholtz_filter(project(iso_filter(z, e), Orient), Orient)
        orient.rename('Orientation vector field', 'label')
        theta = project(operators.atan_2(orient[1], orient[0]), FunctionSpace(self.mesh, 'CG', 1))
        theta.rename('Material axial field', 'label')
        normrized_orient = project(as_vector((cos(theta), sin(theta))), Orient)
        normrized_orient.rename('NormalizedVectorField', 'label')

        offset = 0.01
        Density = FunctionSpace(self.mesh, 'CG', 1)
        density = heviside_filter(helmholtz_filter(r, Density), Density, offset=offset)
        density.rename('Relatively density field', 'label')

        V = VectorFunctionSpace(self.mesh, 'CG', 1)
        v = TrialFunction(V)
        dv = TestFunction(V)
        vh = Function(V, name='Displacement vector field')

        Q = rotated_lamina_stiffness_inplane(
            self.material_parameters['E1'],
            self.material_parameters['E2'],
            self.material_parameters['G12'],
            self.material_parameters['nu12'],
            theta
        )
        Q_reduce = Q*density

        bcs = []
        for i in range(len(self.displacement_boundaries_sub0)):
            bcs.append(DirichletBC(V.sub(0), self.applied_displacements_sub0[i], self.displacement_boundaries_sub0[i]))
        for i in range(len(self.displacement_boundaries_sub1)):
            bcs.append(DirichletBC(V.sub(1), self.applied_displacements_sub1[i], self.displacement_boundaries_sub1[i]))

        a = inner(stress(Q_reduce, v), strain(dv))*dx
        L = dot(self.applied_loadvectors[0], dv)*ds(1)
        if len(self.applied_loadvectors) > 1:
            for i in range(len(self.load_conditions)-1):
                L += dot(self.applied_loadvectors[i+1], dv)*ds(i+2)
        else:
            pass
        solve(a==L, vh, bcs)
        strain_energy = 0.5*inner(stress(Q_reduce, vh), strain(vh))*dx
        cost = assemble(strain_energy)

        sigma = project(stress(Q_reduce, vh), TensorFunctionSpace(self.mesh, 'CG', 1))
        sigma.rename('Stress tensor field', 'label')
        eps = project(strain(vh), TensorFunctionSpace(self.mesh, 'CG', 1))
        eps.rename('Strain tensor field', 'label')

        self.files['Displacement'].write(vh)
        self.files['Stress'].write(sigma)
        self.files['Strain'].write(eps)
        self.files['Orient'].write(orient)
        self.files['Dens'].write(density)
        self.files['Orientpipe'] << normrized_orient
        self.files['Denspipe'] << density

        return cost

class RelativelyDensityResponce(torch_fenics.FEniCSModule):
    """# RelativelDensityResponce
    
    Relatively density responce.

    Attributes:
        mesh (fenics.Mesh): mesh for FE analysis.
    """
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
    def input_templates(self):
        return Function(FunctionSpace(self.mesh, 'CG', 1))
    def solve(self, r):
        """# solve

        calcuate relatively density from given design parameters.

        Args:
            r: Relatively density field

        Returns:
            f: Relatively density
        """
        rho_0_f = project(Constant(1.0), FunctionSpace(self.mesh, 'CG', 1))
        rho_0 = assemble(rho_0_f*dx)
        rho_f = heviside_filter(helmholtz_filter(r, FunctionSpace(self.mesh, 'CG', 1)), FunctionSpace(self.mesh, 'CG', 1))
        rho = assemble(rho_f*dx)
        return rho/rho_0

class VolumeConstraint():
    """# VolumeConstraint
    """
    def __init__(self, responce):
        self.responce = responce
    def __eval(self, x):
        _z, _e, _r = np.split(x, 3)
        r = torch.tensor([_r.tolist()], requires_grad=True, dtype=torch.float64)
        return self.responce(r).detach().numpy().copy()
    def __grad(self, x):
        _z, _e, _r = np.split(x, 3)
        z = torch.tensor([_z.tolist()], requires_grad=True, dtype=torch.float64)
        e = torch.tensor([_e.tolist()], requires_grad=True, dtype=torch.float64)
        r = torch.tensor([_r.tolist()], requires_grad=True, dtype=torch.float64)
        f = self.responce(r)
        f.backward()
        dfdz, dfde, dfdr = 0*z, 0*e, r.grad
        return torch.cat((dfdz, dfde, dfdr), 1).squeeze().detach().numpy().copy()
    def template(self, x, grad, target):
        grad[:] = self.__grad(x)
        return float(self.__eval(x) - target)

class Optimizer():
    """# Optimizer
    """
    def __init__(self):
        self.mesh = None
        self.load_conditions = None
        self.applied_loadvectors = None
        self.bcs_0 = None
        self.bcs_1 = None
        self.applied_displacements_0 = None
        self.applied_displacements_1 = None
        self.material_parameters = None
        self.files = None
        self.target = None
        self.problem = None

    def __template(self, x, grad):
        _z, _e, _r = np.split(x, 3)
        z = torch.tensor([_z.tolist()], requires_grad=True, dtype=torch.float64)
        e = torch.tensor([_e.tolist()], requires_grad=True, dtype=torch.float64)
        r = torch.tensor([_r.tolist()], requires_grad=True, dtype=torch.float64)
        _cost = self.problem(z, e, r)
        _cost.backward()
        dcdz = z.grad
        dcde = e.grad
        dcdr = r.grad
        dcdx = torch.cat((dcdz, dcde, dcdr), 1).squeeze().detach().numpy().copy()
        cost = float(_cost.detach().numpy().copy())
        grad[:] = dcdx
        f = open('optimization_log.csv', 'a')
        f.write("{}\n".format(cost))
        f.close()
        print(cost)
        return cost

    def set_mesh(self, mesh):
        self.mesh = mesh
        self.count_vertices = mesh.num_vertices()
        pass

    def set_bcs(self, boundaries_sub0, boundaries_sub1, applied_displacements_sub0, applied_displacements_sub1):
        self.bcs_0 = boundaries_sub0
        self.bcs_1 = boundaries_sub1
        self.applied_displacements_0 = applied_displacements_sub0
        self.applied_displacements_1 = applied_displacements_sub1
        pass

    def set_loading(self, boundaries, applied_load):
        self.load_conditions = boundaries
        self.applied_loadvectors = applied_load
        pass

    def set_material(self, materials):
        self.material_parameters = materials
        pass

    def set_working_dir(self, files):
        self.files = files
        pass

    def set_target(self, target):
        self.target = target
        pass

    def initialize(self):
        self.problem = CoreProcess(self.mesh,
                                   self.load_conditions,
                                   self.applied_loadvectors,
                                   self.bcs_0,
                                   self.bcs_1,
                                   self.applied_displacements_0,
                                   self.applied_displacements_1,
                                   self.material_parameters,
                                   self.files)
        pass

    def run(self, x0):
        constraint = VolumeConstraint(RelativelyDensityResponce(self.mesh))
        solver = nl.opt(nl.LD_MMA, self.count_vertices*3)
        solver.set_min_objective(self.__template)
        solver.add_inequality_constraint(lambda x, grad: constraint.template(x, grad, self.target), 1e-8)
        solver.set_lower_bounds(-1.0)
        solver.set_upper_bounds(1.0)
        solver.set_xtol_rel(1e-10)
        solver.set_param('verbosity', 2)
        solver.set_maxeval(100)
        x = solver.optimize(x0)
        pass

        

    
        







        

