from dolfin import *
from fenics_shells import *

mesh = UnitSquareMesh(32, 32)

element = MixedElement([VectorElement("Lagrange", triangle, 2),
                        FiniteElement("Lagrange", triangle, 1),
                        FiniteElement("N1curl", triangle, 1),
                        FiniteElement("N1curl", triangle, 1)])

Q = ProjectedFunctionSpace(mesh, element, num_projected_subspaces=2)
Q_F = Q.full_space

q_ = Function(Q_F)
theta_, w_, R_gamma_, p_ = split(q_)
q = TrialFunction(Q_F)
q_t = TestFunction(Q_F)

E = Constant(10920.0)
nu = Constant(0.3)
kappa = Constant(5.0/6.0)
t = Constant(0.001)

k = sym(grad(theta_))

D = (E*t**3)/(12.0*(1.0 - nu**2))
psi_b = 0.5*D*((1.0 - nu)*tr(k*k) + nu*(tr(k))**2)
psi_s = ((E*kappa*t)/(4.0*(1.0 + nu)))*inner(R_gamma_, R_gamma_)
f = Constant(1.0)
W_ext = inner(f*t**3, w_)*dx
gamma = grad(w_) - theta_

dSp = Measure('dS', metadata={'quadrature_degree': 1})
dsp = Measure('ds', metadata={'quadrature_degree': 1})

n = FacetNormal(mesh)
t = as_vector((-n[1], n[0]))

#inner_e = lambda x, y: (inner(x, t)*inner(y, t))('+')*dSp + \
#                       (inner(x, t)*inner(y, t))('-')*dSp + \
#                       (inner(x, t)*inner(y, t))*dsp

Pi_R = inner_e(gamma - R_gamma_, p_)

Pi = psi_b*dx + psi_s*dx + Pi_R - W_ext

dPi = derivative(Pi, q_, q_t)
J = derivative(dPi, q_, q)
A, b = assemble(Q, J, -dPi)
def all_boundary(x, on_boundary):
    return on_boundary

bcs = [DirichletBC(Q, Constant((0.0, 0.0, 0.0)), all_boundary)]

for bc in bcs:
    bc.apply(A, b)

q_p_ = Function(Q)
solver = PETScLUSolver("mumps")
solver.solve(A, q_p_.vector(), b)
reconstruct_full_space(q_, q_p_, J, -dPi)

save_dir = "results/RM_shell"
theta_h, w_h, R_gamma_h, p_h = q_.split()
fields = {"theta": theta_h, "w": w_h, "R_gamma": R_gamma_h, "p": p_h}
for name, field in fields.items():
    field.rename(name, name)
    field_file = XDMFFile("%s/%s.xdmf" % (save_dir, name))
    field_file.write(field)