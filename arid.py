from dolfin import *
import numpy as np
import random

c = 10.0    #conversion of water uptake to plant growth
gmax = 0.05 #maximum specific water uptake
k1 = 5.0    #half saturation constant of specific h2o uptake
d = 0.3     # 0.0-0.5 specific loss of plant density due to mortality
Dp = 0.2    #plant dispersal
alpha = 0.2 #maximum infiltration rate
k2 = 5.0    #saturation constant of water infiltration
W0 = 0.2    #water infiltration rate in absence of plants
rw = 0.2    #specific soil water loss - evaporation+drainage
Dw = 0.1    #diffusion coeff of soil water
Do = 50.0   #diffusion coeff for surface water
R = 1.3     #rainfall
Lx = 400
Ly = 400
Nx = 200
Ny = 200
#define periodic boundary conditions
class PeriodicBC(SubDomain):
    def inside(self, x, on_boundary):
        return bool((near(x[0], 0) or near(x[1], 0)) and 
                (not ((near(x[0], 0) and near(x[1], Ly)) or 
                        (near(x[0], Lx) and near(x[1], 0)))) and on_boundary)
    def map(self, x, y):
        if near(x[0], Lx) and near(x[1], Lx):
            y[0] = x[0] - Lx
            y[1] = x[1] - Ly
        elif near(x[0], Lx):
            y[0] = x[0] - Lx
            y[1] = x[1]
        else:   # near(x[1], Ly)
            y[0] = x[0]
            y[1] = x[1] - Ly
class InitialConditions(Expression):
    def __init__(self):
        random.seed(2 + MPI.rank(mpi_comm_world()))
    def eval(self, values, x):
        values[0] = 100.0*float(np.random.rand() > .99)
        values[1] = R/rw
        values[2] = R/(alpha*W0)
    def value_shape(self):
        return (3,)
pbc = PeriodicBC()
bcs = [] #no other boundary conditions
#mesh and function spaces
mesh = RectangleMesh(0, 0, Lx, Ly, Nx, Ny)
Y = FunctionSpace(mesh , "CG", 1, constrained_domain=pbc)
Z = MixedFunctionSpace([Y, Y, Y])

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

#test functions + previous iter. solns
z = Function(Z); (p, w, o) = split(z)
z0 = Function(Z); (p0,w0,o0) = split(z0)
(pt, wt, ot) = TestFunctions(Z)
du = TrialFunction(Z)
z_init = InitialConditions()
z.interpolate(z_init)
z0.interpolate(z_init)
dt = 0.05

P = p*pt*dx - p0*pt*dx - dt*(c*gmax*(w/(w+k1))*p - d*p)*pt*dx + dt*Dp*dot(grad(p),grad(pt))*dx 
W = w*wt*dx - w0*wt*dx - dt*(alpha*o*((p+(k2*W0))/(p+k2)) - gmax*(w/(w+k1))*p - rw*w)*wt*dx + dt*Dw*dot(grad(w),grad(wt))*dx
O = o*ot*dx - o0*ot*dx - dt*(R - alpha*o*((p+(k2*W0))/(p+k2)))*ot*dx + dt*Do*dot(grad(o),grad(ot))*dx

F = P + W + O
J = derivative(F,z, du)
t = 0.0
T = 150
problem = NonlinearVariationalProblem(F, z, bcs, J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['maximum_iterations'] = 100

ntime = T/dt
xavg = np.zeros(ntime+1)
savg = np.zeros(ntime+1)
yavg = np.zeros(ntime+1)

file = XDMFFile(mpi_comm_world(), "outarid.xdmf")
timestep = 0
while (t<=T):
    t += dt
    z0.vector()[:] = z.vector()
    #solve(F == 0, z, bcs)
    solver.solve()

    xavg[timestep] = assemble(z.split()[0]*dx(mesh)) / assemble(Constant(1)*dx(mesh))
    savg[timestep] = assemble(z.split()[1]*dx(mesh)) / assemble(Constant(1)*dx(mesh))
    yavg[timestep] = assemble(z.split()[2]*dx(mesh)) / assemble(Constant(1)*dx(mesh))

    timestep += 1

    if timestep % 50 == 0:
        file << (z.split()[0], t)
np.savetxt('aridts.out', (xavg,savg,yavg))





