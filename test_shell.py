from anisogrid.shell import Optimizer
from fenics import *
from dolfin import *
from fenics_adjoint import *
import numpy as np

mesh = RectangleMesh(Point(0, 0), Point(25,25), 50, 50)
N = mesh.num_vertices()
print("DOF: {}".format(N*5))

class All_boundaries(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class Left_bottom(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 2 and x[1] < 2

class Left_top(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 2 and x[1] > 23

class Right_top(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 23 and x[1] > 23 

class Right_bottom(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 23 and x[1] < 2

class Center(SubDomain):
    def inside(self, x, on_boundary):
        return 11 < x[0] < 14 and 11 < x[1] < 14

w_boundaries = [Left_bottom(), Left_top(), Right_bottom(), Right_top(), Center()]
applied_w = [Constant(0), Constant(0), Constant(0), Constant(0), Constant(-2)]
rotation_boundaries_sub0 = []
rotation_boundaries_sub1 = []
applied_rotation_sub0 = []
applied_rotation_sub1 = []

material = {'E1': 7500, 'E2': 500, 'nu12':0.33, 'G12': 400, 'G23': 300}
path = 'results/UniformLoading/80_ex2/'
files = {'Displacement': XDMFFile('{}displacement.xdmf'.format(path)),
         'Orient': XDMFFile('{}orient.xdmf'.format(path)),
         'Orientpipe': File('{}orient.xml'.format(path)),
         'Denspipe': File('{}dens.xml'.format(path)),
         'Dens': XDMFFile('{}dens.xdmf'.format(path))
        }

z0 = np.ones(N)
e0 = np.zeros(N)
r0 = np.zeros(N)
x0 = np.concatenate([z0, e0, r0])

opt = Optimizer()
opt.set_mesh(mesh)
opt.set_bcs(w_boundaries, applied_w, rotation_boundaries_sub0, applied_rotation_sub0, rotation_boundaries_sub1, applied_rotation_sub1)
opt.set_material(material)
opt.set_working_dir(files)
opt.set_target(0.9, coffSigmoid=50)
opt.initialize()
opt.run(x0, max_itr=50)