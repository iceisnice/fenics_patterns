from dolfin import *
import random
import numpy as np
from math import sqrt
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

eps_1 = 2.0e-5
eps_2 = 1.0e-5
k = 0.065
eff = 0.05
class InitialConditions(Expression):
    def __init__(self):
        random.seed(2 + MPI.rank(mpi_comm_world()))
    def eval(self, values, x):
        values[0] = (eff/(k + (2.0*eff))) + 1.0*float(np.random.rand() > .98)
        values[1] = sqrt(k + eff)#*float(np.random.rand() > .99)
    def value_shape(self):
        return (2,)
mesh = UnitSquareMesh(64,64)
Y = FunctionSpace(mesh , "CG", 1)
Z = Y*Y


z = Function(Z); (p, h) = split(z)
z0 = Function(Z); (p0,h0) = split(z0)
(pt, ht) = TestFunctions(Z)
ddz = TrialFunction(Z)
z_init = InitialConditions()
z.interpolate(z_init)
z0.interpolate(z_init)



dt = 0.01; T = 30; t = 0.0;

P = (p - p0)*pt*dx + dt*eps_1*inner(grad(p),grad(pt))*dx - dt*eff*(1-p)*pt*dx + dt*p*h*h*pt*dx 
H = (h - h0)*ht*dx + dt*eps_2*inner(grad(h),grad(ht))*dx - dt*p*h*h*ht*dx + dt*(k + eff)*h*ht*dx


F = P + H
J = derivative(F,z,ddz)
bcs = []
ntime = T/dt
xavg = np.zeros(ntime+1)
savg = np.zeros(ntime+1)
file = File("outgs1.xdmf")
file2 = XDMFFile(mpi_comm_world(), "outgs2.xdmf")
timestep = 0

problem = NonlinearVariationalProblem(F, z, bcs, J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters

prm['newton_solver']['maximum_iterations'] = 200

while (t<=T):
    t += dt
    z0.vector()[:] = z.vector()
    solver.solve()
    xavg[timestep] = assemble(z.split()[0]*dx(mesh)) / assemble(Constant(1)*dx(mesh))
    savg[timestep] = assemble(z.split()[1]*dx(mesh)) / assemble(Constant(1)*dx(mesh))
    timestep += 1
    print timestep
    if timestep % 25 == 0:
        file << (z.split()[0], t)
        file2 << (z.split()[1],t)
np.savetxt('gscotttsout', (xavg,savg))
