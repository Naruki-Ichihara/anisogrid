from anisogrid.tools import Optimizer
from fenics import *
from fenics_adjoint import *
import numpy as np

mesh = RectangleMesh(Point(0,0), Point(50,20), 100, 40)
N = mesh.num_vertices()

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.01 and on_boundary

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 45 and x[1] < 0.1 and on_boundary

class Loading(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 19.9 and x[0] < 5. and on_boundary

displacement_boundaries_0 = [Left()]
displacement_boundaries_1 = [Right()]
applied_disp_0 = [Constant(0)]
applied_disp_1 = [Constant(0)]

loading_boundaries = [Loading()]
applied_loads = [Constant((0, -100))]

material = {'E1': 3600, 'E2': 600, 'nu12':0.33, 'G12': 500}
path = 'results/example1/'
files = {'Displacement': XDMFFile('{}displacement.xdmf'.format(path)),
         'Stress': XDMFFile('{}stress.xdmf'.format(path)),
         'Strain': XDMFFile('{}strain.xdmf'.format(path)),
         'Orient': XDMFFile('{}orient.xdmf'.format(path)),
         'Orientpipe': File('{}orient.xml'.format(path)),
         'Denspipe': File('{}dens.xml'.format(path)),
         'Dens': XDMFFile('{}dens.xdmf'.format(path))
        }

z0 = np.ones(N)
e0 = np.zeros(N)
r0 = np.ones(N)
x0 = np.concatenate([z0, e0, r0])

opt = Optimizer()
opt.set_mesh(mesh)
opt.set_bcs(displacement_boundaries_0, displacement_boundaries_1, applied_disp_0, applied_disp_1)
opt.set_loading(loading_boundaries, applied_loads)
opt.set_material(material)
opt.set_working_dir(files)
opt.set_target(0.8)
opt.initialize()
opt.run(x0)