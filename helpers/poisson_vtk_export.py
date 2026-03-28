#!/usr/bin/env python3
# poisson_vtk_export.py
# Export Poisson solution in multiple formats for better compatibility

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

def solve_and_export_multiple_formats():
    """Solve Poisson equation and export in multiple formats"""
    
    print("=== SOLVING POISSON EQUATION ===")
    
    # 1. Create mesh and function space
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", 1))
    print("✓ Mesh and function space created")

    # 2. Define boundary conditions
    uD = fem.Function(V)
    uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)
    print("✓ Boundary conditions defined")

    # 3. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(-6))
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    print("✓ Variational problem defined")

    # 4. Solve
    A = fem_petsc.assemble_matrix(fem.form(a), bcs=[bc])
    A.assemble()

    b = fem_petsc.assemble_vector(fem.form(L))
    fem_petsc.apply_lifting(b, [fem.form(a)], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, [bc])

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    uh = fem.Function(V)
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    print("✓ System solved")

    # 5. Export in multiple formats
    print("\n=== EXPORTING IN MULTIPLE FORMATS ===")
    
    # Export 1: XDMF (original)
    try:
        with io.XDMFFile(domain.comm, "poisson_solution.xdmf", "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(uh)
        print("✓ XDMF export successful")
    except Exception as e:
        print(f"✗ XDMF export failed: {e}")
    
    # Export 2: VTX (new VTK-based format)
    try:
        with io.VTXWriter(domain.comm, "poisson_solution.bp", [uh]) as vtx:
            vtx.write(0.0)
        print("✓ VTX export successful")
    except Exception as e:
        print(f"✗ VTX export failed: {e}")
    
    # Export 3: Try legacy VTK if available
    try:
        # Note: This might not be available in all DOLFINx versions
        with io.VTKFile(domain.comm, "poisson_solution.pvd", "w") as vtk:
            vtk.write_function(uh)
        print("✓ VTK export successful")
    except Exception as e:
        print(f"✗ VTK export failed: {e}")
    
    # Export 4: Raw data for manual visualization
    try:
        coords = V.tabulate_dof_coordinates()
        values = uh.x.array
        
        # Save as numpy arrays
        np.savez('poisson_solution_raw.npz', 
                coordinates=coords, 
                values=values,
                description="Poisson solution: coordinates and values")
        
        # Save as simple text file
        with open('poisson_solution.txt', 'w') as f:
            f.write("# x y u(x,y)\n")
            for i in range(len(coords)):
                f.write(f"{coords[i,0]:.6f} {coords[i,1]:.6f} {values[i]:.6f}\n")
        
        print("✓ Raw data export successful")
        print(f"  - NumPy format: poisson_solution_raw.npz")
        print(f"  - Text format: poisson_solution.txt")
        
    except Exception as e:
        print(f"✗ Raw data export failed: {e}")

    # 6. Solution statistics
    solution_max = np.max(uh.x.array)
    solution_min = np.min(uh.x.array)
    print(f"\n✓ Solution range: [{solution_min:.6f}, {solution_max:.6f}]")
    
    # 7. List all generated files
    import os
    print(f"\n=== GENERATED FILES ===")
    extensions = ['.xdmf', '.h5', '.bp', '.pvd', '.vtu', '.npz', '.txt']
    for ext in extensions:
        files = [f for f in os.listdir('.') if f.endswith(ext) and 'poisson' in f]
        for file in files:
            size = os.path.getsize(file)
            print(f"  {file}: {size} bytes")

if __name__ == "__main__":
    solve_and_export_multiple_formats()
