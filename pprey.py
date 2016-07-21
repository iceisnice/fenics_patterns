from dolfin import *
import random
import numpy as np
from mshr import *

domain =   Rectangle(dolfin.Point(0., 0.), dolfin.Point(400., 400.)) \
         - Rectangle(dolfin.Point(100., 100.), dolfin.Point(110., 170.)) \
         - Circle(dolfin.Point(300., 320.), 30.)

a = 3.5; b = 1.2; c = 5;
heq = (1/((a*c)-c))
peq = ((1-heq)*(1+(c*heq))/c)
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

def outer_boundary(x):
    return (near(x[0], 0.) or near(x[0], 400.) or near(x[1], 0.) or near(x[1], 400.))
def obstacles(x, on_boundary):
    return (on_boundary and not outer_boundary(x))  

#class InitialConditions(Expression):
#    def __init__(self):
#        random.seed(2 + MPI.rank(mpi_comm_world()))
#    def eval(self, values, x):
#        values[0] = ((1-heq)*(1+(c*heq))/c) + (random.random() - peq)
#        values[1] = heq + (random.random() - heq)
#    def value_shape(self):
#        return (2,)
class InitialConditions(Expression):
    def __init__(self):
        random.seed(2 + MPI.rank(mpi_comm_world()))
    def eval(self, values, x):
        values[0] = 1.0*float(np.random.rand() > .995)
        values[1] = 1.0
    def value_shape(self):
        return (2,)

mesh = generate_mesh(domain, 200)

Y = FunctionSpace(mesh , "CG", 1)
Z = Y*Y

z = Function(Z); (p, h) = split(z)
z0 = Function(Z); (p0,h0) = split(z0)
(pt, ht) = TestFunctions(Z)
ddz = TrialFunction(Z)
z_init = InitialConditions()
z.interpolate(z_init)
z0.interpolate(z_init)



dt = 0.1; T = 300.0; t = 0.0;

P = (p - p0)*pt*dx - dt*(p/(a*b))*(((a*c*h)/(1+(c*h)))-1)*pt*dx + dt*2*inner(grad(p),grad(pt))*dx
H = (h - h0)*ht*dx - dt*h*(1-h)*ht*dx + dt*((c*p*h)/(1+(c*h)))*ht*dx + dt*inner(grad(h),grad(ht))*dx

F = P + H
J = derivative(F,z,ddz)
bcs = DirichletBC(Z, Constant((0.0,0.0)), obstacles)


file = XDMFFile(mpi_comm_world(), "outpred.xdmf")
preyfile = XDMFFile(mpi_comm_world(), "outprey.xdmf")
timestep = 0
ntime = int(T/dt)
xavg = np.zeros(ntime+1)
savg = np.zeros(ntime+1)
yavg = np.zeros(ntime+1)
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
    if timestep % 50 == 0:
        file << (z.split()[0], t)
        preyfile << (z.split()[1],t)
             
np.savetxt('preyfile.out', (xavg,savg))
