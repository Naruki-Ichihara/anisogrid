from dolfin import *
from fenics_shells import *

# Generate mesh
mesh = UnitSquareMesh(64, 64)

# Define finite element spaces.
# 2nd order CG element for the rotation field
# 1st order CG element for the membrane displacement field
# 1st order edge element (Nedelec element) for the reduced shear strain and Lagrange multiplier

element = MixedElement([VectorElement("Lagrange", triangle, 2),
                        FiniteElement("Lagrange", triangle, 1),
                        FiniteElement("N1curl", triangle, 1),
                        FiniteElement("N1curl", triangle, 1)])

U = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
U_F = U.full_space

u_ = Function(U_F)
theta_, w_, R_gamma_, p_ = split(u_)
u = TrialFunction(U_F)
u_t = TestFunction(U_F)

E = Constant(10920.0)
nu = Constant(0.3)
kappa = Constant(5.0/6.0)
t = Constant(0.0001)

# bending strain tensor can be expressed in terms of the rotation field
k = sym(grad(theta_))

D = (E*t**3)/(24.0*(1.0 - nu**2))

# Membrane strain energy is
psi_M = D*((1.0 - nu)*tr(k*k) + nu*(tr(k))**2)
# reduced shear strain energy is
psi_T = ((E*kappa*t)/(4.0*(1.0 + nu)))*inner(R_gamma_, R_gamma_)
# External force is
f = Constant(1.0)
W_ext = inner(f*t**3, w_)*dx

# (non-reduced) shear strain can be expressed in terms of the grad of membrane displacement
gamma = grad(w_) - theta_

# We require that the shear strain equal to the reduced shear strain
# We enforce this constraints using Lagrange method with multiplier p
L_R = inner_e(gamma - R_gamma_, p_)

# Then, total system is
L = psi_M*dx + psi_T*dx + L_R - W_ext

F = derivative(L, u_, u_t)
J = derivative(F, u_, u)

A, b = assemble(U, J, -F)

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

# Simply supported boundary conditions.
bcs = [DirichletBC(U.sub(1), Constant(0.0), all_boundary),
       DirichletBC(U.sub(0).sub(0), Constant(0.0), top),
       DirichletBC(U.sub(0).sub(0), Constant(0.0), bottom),
       DirichletBC(U.sub(0).sub(1), Constant(0.0), left),
       DirichletBC(U.sub(0).sub(1), Constant(0.0), right)]

for bc in bcs:
    bc.apply(A, b)

u_p_ = Function(U)
solver = PETScLUSolver("mumps")
solver.solve(A, u_p_.vector(), b)
reconstruct_full_space(u_, u_p_, J, -F)

save_dir = "results/RM_shell"
theta, w, R_gamma, p = u_.split()
fields = {"theta": theta, "w": w, "R_gamma": R_gamma, "p": p}
for name, field in fields.items():
    field.rename(name, name)
    field_file = XDMFFile("%s/%s.xdmf" % (save_dir, name))
    field_file.write(field)