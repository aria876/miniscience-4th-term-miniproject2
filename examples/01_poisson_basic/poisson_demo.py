# poisson_demo_v0.9.0_final.py
# Working Poisson solver for DOLFINx v0.9.0
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

# 1. Create mesh and function space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
V = fem.functionspace(domain, ("Lagrange", 1))

# 2. Define boundary conditions
uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

# 3. Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(-6))
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# 4. Assemble the linear system
A = fem_petsc.assemble_matrix(fem.form(a), bcs=[bc])
A.assemble()

b = fem_petsc.assemble_vector(fem.form(L))
fem_petsc.apply_lifting(b, [fem.form(a)], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem_petsc.set_bc(b, [bc])

# 5. Create solver and solve
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

uh = fem.Function(V)
solver.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward()

# 6. Save the solution
with io.XDMFFile(domain.comm, "poisson_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

print("SUCCESS: Solution saved to poisson_solution.xdmf")
print(f"Solution range: [{np.min(uh.x.array):.6f}, {np.max(uh.x.array):.6f}]")
