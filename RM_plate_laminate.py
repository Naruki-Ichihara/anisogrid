from dolfin import *
from fenics_shells import *
import numpy as np

# Generate mesh
mesh = UnitSquareMesh(64, 64)

# Define finite element spaces.
# Ist order CG element for the in-plane displacement
# 2nd order CG element for the rotation field
# 1st order CG element for the bending displacement field
# 1st order edge element (Nedelec element) for the reduced shear strain and Lagrange multiplier

element = MixedElement([VectorElement("Lagrange", triangle, 1),
                        VectorElement("Lagrange", triangle, 2),
                        FiniteElement("Lagrange", triangle, 1),
                        FiniteElement("N1curl", triangle, 1),
                        FiniteElement("N1curl", triangle, 1)])

U = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
U_F = U.full_space

u_ = Function(U_F)
v_, theta_, w_, R_gamma_, p_ = split(u_)
u = TrialFunction(U_F)
u_t = TestFunction(U_F)

thetas = [np.deg2rad(45), np.deg2rad(-45)]
E1 = 15
E2 = 1.0
G12 = 0.5
nu12 = 0.25
G23 = 0.4
h = 1
t = Constant(1)

n_layers= len(thetas)
hs = h*np.ones(n_layers)/n_layers
A, B, D = laminates.ABD(E1, E2, G12, nu12, hs, thetas)
Fs = laminates.F(G12, G23, hs, thetas)

# membrane strain tensor for the von-Karman plate

# membrane strain energy is
def psi_N(v, w):
    e = sym(grad(v)) + 0.5*outer(grad(w), grad(w))
    ev = strain_to_voigt(e)
    Ai = project(A, TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3)))
    return .5*dot(Ai*ev, ev)

# bending strain tensor can be expressed in terms of the rotation field
def psi_M(theta):
    k = sym(grad(theta))
    kv = strain_to_voigt(k)
    Di = project(D, TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3)))
    return  .5*dot(Di*kv, kv)

# reduced shear strain energy is
def psi_T(gamma):
    Fi = project(Fs, TensorFunctionSpace(mesh, 'CG', 1, shape=(2,2)))
    return .5*dot(Fi*gamma, gamma)

# Coupling strain energy is

def psi_MN(theta, v, w):
    k = sym(grad(theta))
    ki = strain_to_voigt(k)
    e = sym(grad(v)) + 0.5*outer(grad(w), grad(w))
    ei = strain_to_voigt(e)
    Bi = project(B, TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3)))
    return dot(Bi*ki, ei)

# External force is
facets = MeshFunction('size_t', mesh, 1)
facets.set_all(0)
class Center(SubDomain):
    def inside(self, x, on_boundary):
        return 0.4 < x[0] < 0.5 and 0.4 < x[1] < 0.5
Center().mark(facets, 1)
dx_c = Measure('dx', subdomain_data=facets)
f = Constant(1.0)
W_ext = inner(f, w_)*dx

# (non-reduced) shear strain can be expressed in terms of the grad of membrane displacement
gamma = grad(w_) - theta_

# We require that the shear strain equal to the reduced shear strain
# We enforce this constraints using Lagrange method with multiplier p
L_R = inner_e(gamma - R_gamma_, p_)

# Then, total system is
L = psi_M(theta_)*dx + psi_T(R_gamma_)*dx + psi_N(v_, w_)*dx + psi_MN(theta_, v_, w_)*dx + L_R

F = derivative(L, u_, u_t)
J = derivative(F, u_, u)

A_, b = assemble(U, J, -F)

# Setup the SubDomains for boundary conditions
def all_boundary(x, on_boundary):
    return on_boundary

def left(x, on_boundary):
    return on_boundary and near(x[0], 0.0)

def right(x, on_boundary):
    return on_boundary and near(x[0], 1.0)

def bottom(x, on_boundary):
    return on_boundary and near(x[1], 0.0)

def top(x, on_boundary):
    return on_boundary and near(x[1], 1.0)

def disp(x, on_boundary):
    return 0.9 < x[0] and x[1] < 0.1

# Simply supported boundary conditions.
bcs = [DirichletBC(U.sub(2), Constant(0.0), left),
       DirichletBC(U.sub(2), Constant(0.0), top),
       DirichletBC(U.sub(1).sub(0), Constant(0.0), top),
       DirichletBC(U.sub(1).sub(1), Constant(0.0), left),
       DirichletBC(U.sub(2), Constant(-1), disp)]

for bc in bcs:
    bc.apply(A_, b)

u_p_ = Function(U)
solver = PETScLUSolver("mumps")
solver.solve(A_, u_p_.vector(), b)
reconstruct_full_space(u_, u_p_, J, -F)

save_dir = "results/RM_shell_laminate"
v, theta, w, R_gamma, p = u_.split()
# Compute total strain energy
E = assemble(psi_N(v, w)*dx + psi_MN(theta, v, w)*dx + psi_M(theta)*dx + psi_T(R_gamma)*dx)
print(E)
fields = {"displacement": v, "theta": theta, "w": w, "R_gamma": R_gamma, "p": p}
for name, field in fields.items():
    field.rename(name, name)
    field_file = XDMFFile("%s/%s.xdmf" % (save_dir, name))
    field_file.write(field)