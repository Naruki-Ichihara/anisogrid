#!opt/conda/envs/project/bin/python
# -*- coding: utf-8 -*-
"""Part of anisogrid. The toolkit for the anisotropic topology optimization"""
from ufl.core.ufl_type import determine_num_ops
from ufl.functionspace import MixedFunctionSpace
from fenics_shells import *
from fenics import *
from fenics_adjoint import *
import numpy as np
from ufl import operators, transpose
from torch_fenics import *
import torch
import nlopt as nl
from tqdm import tqdm

class CustomSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self,
                              PETScKrylovSolver(), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)

        PETScOptions.set("ksp_type", "mumps")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("pc_type", "ilu")

        self.linear_solver().set_from_options()

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

def heviside_filter(u, U, a=50, offset=0.5):
    """# heviside_filter
    
    Apply the heviside function (approximate with sigmoid function)

    Args:
        u (fenics.function): target function
        U (fenics.FunctionSpace): Functionspace of target function
        a (float): coefficient. a>50 -> Step. a=3 -> S-shape

    Returns:
        v (fenics.function): filterd function

    Note:
    
    """
    val = (1 / (1 + exp(-a*u)))*(1-offset) + offset
    return project(val, U)

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
    def __init__(self, mesh, w_boundaries, applied_w, rotation_boundaries_sub0, applied_rotation_sub0, rotation_boundaries_sub1, applied_rotation_sub1, material_parameters, files, a):
        super().__init__()
        self.mesh = mesh
        self.w_boundaries = w_boundaries
        self.applied_w = applied_w
        self.rotation_boundaries_sub0 = rotation_boundaries_sub0
        self.applied_rotation_sub0 = applied_rotation_sub0
        self.rotation_boundaries_sub1 = rotation_boundaries_sub1
        self.applied_rotation_sub1 = applied_rotation_sub1
        self.material_parameters = material_parameters
        self.files = files
        self.a = a
    
    # membrane strain energy is
    def psi_N(self, v, v_t, A):
        e = sym(grad(v))
        ev = strain_to_voigt(e)
        e_t = sym(grad(v_t))
        ev_t = strain_to_voigt(e_t)
        Ai = project(A, TensorFunctionSpace(self.mesh, 'CG', 1, shape=(3,3)))
        return .5*dot(Ai*ev, ev_t)

    # bending strain tensor can be expressed in terms of the rotation field
    def psi_M(self, theta, theta_t, D):
        k = sym(grad(theta))
        kv = strain_to_voigt(k)
        k_t = sym(grad(theta_t))
        kv_t = strain_to_voigt(k_t)
        Di = project(D, TensorFunctionSpace(self.mesh, 'CG', 1, shape=(3,3)))
        return  .5*dot(Di*kv, kv_t)

    # reduced shear strain energy is
    def psi_T(self, gamma, gamma_t, Fs):
        Fi = project(Fs, TensorFunctionSpace(self.mesh, 'CG', 1, shape=(2,2)))
        return .5*dot(Fi*gamma, gamma_t)

    # Coupling strain energy is
    def psi_MN(self, theta, v_t, B):
        k = sym(grad(theta))
        ki = strain_to_voigt(k)
        e_t = sym(grad(v_t))# + 0.5*outer(grad(w), grad(w))
        ei_t = strain_to_voigt(e_t)
        Bi = project(B, TensorFunctionSpace(self.mesh, 'CG', 1, shape=(3,3)))
        return dot(Bi*ki, ei_t)
    
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
        bar = tqdm(total=4)
        bar.set_description('Constracting finite element spaces..')
        bar.update(1)
        # Define finite element spaces.
        # Ist order CG element for the in-plane displacement
        # 2nd order CG element for the rotation field
        # 1st order CG element for the bending displacement field
        # 1st order edge element (Nedelec element) for the reduced shear strain and Lagrange multiplier

        elements = MixedElement([VectorElement("Lagrange", triangle, 1),
                                VectorElement("Lagrange", triangle, 2),
                                FiniteElement("Lagrange", triangle, 1),
                                FiniteElement("N1curl", triangle, 1),
                                FiniteElement("N1curl", triangle, 1)])
        U = FunctionSpace(self.mesh, elements)

        u = Function(U)
        du = TrialFunction(U)
        ut = TestFunction(U)
        v, theta, w, R_gamma, p = split(u)

        Orient = VectorFunctionSpace(self.mesh, 'CG', 1)
        orient = helmholtz_filter(project(iso_filter(z, e), Orient), Orient)
        orient.rename('Orientation vector field', 'label')
        phi = project(operators.atan_2(orient[1], orient[0]), FunctionSpace(self.mesh, 'CG', 1))
        phi.rename('Material axial field', 'label')
        normrized_orient = project(as_vector((cos(phi), sin(phi))), Orient)
        normrized_orient.rename('NormalizedVectorField', 'label')

        thetas = [np.deg2rad(45), np.deg2rad(-45), phi, np.deg2rad(-45), np.deg2rad(45)]
        E1 = self.material_parameters['E1']
        E2 = self.material_parameters['E2']
        G12 = self.material_parameters['G12']
        nu12 = self.material_parameters['nu12']
        G23 = self.material_parameters['G23']

        A_, B_, D_ = laminates.ABD(E1, E2, G12, nu12, [0.5, 0.5, 1, 0.5, 0.5], thetas)
        Fs = laminates.F(G12, G23, [0.5, 0.5, 1, 0.5, 0.5], thetas)

        Density = FunctionSpace(self.mesh, 'CG', 1)
        density = heviside_filter(helmholtz_filter(r, Density), Density, a=self.a)
        density.rename('Relatively density field', 'label')
        A_ = A_*density
        B_ = B_*density
        D_ = D_*density
        Fs = Fs*density

        # (non-reduced) shear strain can be expressed in terms of the grad of membrane displacement
        gamma = grad(w) - theta

        # We require that the shear strain equal to the reduced shear strain
        # We enforce this constraints using Lagrange method with multiplier p
        L_R = inner_e(gamma - R_gamma, p)

        # Then, total system is
        bar.set_description('Assembling system..')
        bar.update(1)
        F = self.psi_M(theta, theta, D_)*dx + self.psi_N(v, v, A_)*dx + self.psi_T(R_gamma, R_gamma, Fs)*dx + L_R
        dF = derivative(F, u, ut)
        J = derivative(dF, u, du)
        a = derivative(F, u, du)
        A = assemble(J)
        b = assemble(dF)

        bcs = []
        for i in range(len(self.w_boundaries)):
            bcs.append(DirichletBC(U.sub(2), self.applied_w[i], self.w_boundaries[i]))
        for i in range(len(self.rotation_boundaries_sub0)):
            bcs.append(DirichletBC(U.sub(1).sub(0), self.applied_rotation_sub0[i], self.rotation_boundaries_sub0[i]))
        for i in range(len(self.rotation_boundaries_sub1)):
            bcs.append(DirichletBC(U.sub(1).sub(1), self.applied_rotation_sub1[i], self.rotation_boundaries_sub1[i]))


        solver = PETScLUSolver("mumps") 
        for bc in bcs:
            bc.apply(A, b)

        bar.set_description('Solving system..')
        bar.update(1)
        solver.solve(A, u.vector(), b)

        vh, thetah, wh, R_gammah, ph = u.split()
        cost = assemble(self.psi_M(thetah, thetah, D_)*dx + self.psi_N(vh, vh, A_)*dx + self.psi_T(R_gammah, R_gammah, Fs)*dx)
        bar.set_description('Solved system..')
        bar.update(1)

        self.files['Displacement'].write(wh)
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
        self.w_boundaries = None
        self.applied_w = None
        self.rotation_boundaries_sub0 = None
        self.applied_rotation_sub0 = None
        self.rotation_boundaries_sub1 = None
        self.applied_rotation_sub1 = None
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
        """# set_mesh

        Setting the geometry.

        Args:
            mesh: FEniCS mesh instance.

        Returns:
            None
        """
        self.mesh = mesh
        self.count_vertices = mesh.num_vertices()
        pass

    def set_bcs(self, w_boundaries, applied_w, rotation_boundaries_sub0, applied_rotation_sub0, rotation_boundaries_sub1, applied_rotation_sub1):
        """# set_bcs

        Apply displacement boundaries.

        Args:
            boundaries_sub0: list(FEniCS Subdomain instance for the x-axis constraints.)

                            class Clamp(SubDomain):
                                def inside(self, x, on_boundary):
                                    return x[1] < 0 and on_boundary

            boundaries_sub1: list(FEniCS Subdomain instance for the x-axis constraints.)
            applied_displacements_sub0: list(FEniCS constant instace.)
            applied_displacements_sub1: list(FEniCS constant instace.)

        Returns:
            None
        """
        self.w_boundaries = w_boundaries
        self.applied_w = applied_w
        self.rotation_boundaries_sub0 = rotation_boundaries_sub0
        self.applied_rotation_sub0 = applied_rotation_sub0
        self.rotation_boundaries_sub1 = rotation_boundaries_sub1
        self.applied_rotation_sub1 = applied_rotation_sub1
        pass

    def set_material(self, material):
        """# set_material

        Orhogonal anisotropic property was assumed.

        Args:
            material: material = {'E1': 3600, 'E2': 600, 'nu12':0.45, 'G12': 500}
            
        Returns:
            None
        """
        self.material_parameters = material
        pass

    def set_working_dir(self, files):
        """# set_working_dir

        Args:
            files: path = 'results/implant_SOLID/'
                   files = {'Displacement': XDMFFile('{}displacement.xdmf'.format(path)),      
                            'Stress': XDMFFile('{}stress.xdmf'.format(path)),
                            'Strain': XDMFFile('{}strain.xdmf'.format(path)),
                            'Orient': XDMFFile('{}orient.xdmf'.format(path)),
                            'Orientpipe': File('{}orient.xml'.format(path)),
                            'Denspipe': File('{}dens.xml'.format(path)),
                            'Dens': XDMFFile('{}dens.xdmf'.format(path))
        }
            
        Returns:
            None
        """
        self.files = files
        pass

    def set_target(self, target, coffSigmoid=50):
        """# set_target

        Target volume reduction. If this value set to be >1, material density will not update.

        Args: target: float
        coffSigmoid (float): coefficient. >50 -> Step. =3 -> S-shape
        
        Returns:
            None
        
        """
        self.target = target
        self.a = coffSigmoid
        pass

    def initialize(self):
        """# Intialize
        """
        self.problem = CoreProcess(self.mesh,
                                   self.w_boundaries,
                                   self.applied_w,
                                   self.rotation_boundaries_sub0,
                                   self.applied_rotation_sub0,
                                   self.rotation_boundaries_sub1,
                                   self.applied_rotation_sub1,
                                   self.material_parameters,
                                   self.files,
                                   self.a)
        pass

    def run(self, x0, max_itr=100):
        """# run
        """
        constraint = VolumeConstraint(RelativelyDensityResponce(self.mesh))
        solver = nl.opt(nl.LD_MMA, self.count_vertices*3)
        solver.set_max_objective(self.__template)
        solver.add_inequality_constraint(lambda x, grad: constraint.template(x, grad, self.target), 1e-8)
        #solver.set_min_objective(self.__template, None)
        solver.set_lower_bounds(-1.0)
        solver.set_upper_bounds(1.0)
        solver.set_xtol_rel(1e-10)
        solver.set_param('verbosity', 2)
        solver.set_maxeval(max_itr)
        x = solver.optimize(x0)
        pass
