#!/usr/bin/env python3
# nonlinear_poisson_p_laplacian.py
# A demonstration of solving a nonlinear PDE in FEniCSx.
#
# Problem Description:
# We solve the p-Laplacian equation, a nonlinear generalization of the
# standard Poisson equation. The problem is defined as:
#
# -∇ ⋅ (k(u) ∇u) = f
#
# where the conductivity k(u) depends on the solution u itself. Specifically,
# for the p-Laplacian, k(u) = |∇u|^(p-2). This model appears in non-Newtonian
# fluid mechanics, glaciology, and image processing.
#
# For this example, we choose a simpler form to ensure robustness:
# k(u) = 1 + u^2
#
# This makes the equation: -∇ ⋅ ((1 + u^2) ∇u) = f
#
# We will solve this on a unit square with a manufactured analytical solution
# to verify the implementation of the nonlinear solver.

import numpy as np
import ufl
from dolfinx import fem, io, mesh, default_scalar_type
from dolfinx.fem import petsc as fem_petsc
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc
import sympy  # Using SymPy for symbolic computation


def generate_analytical_solution():
    """Use SymPy to manufacture an analytical solution and source term"""
    print("1. Generating analytical solution using SymPy...")
    x, y = sympy.symbols("x[0], x[1]")

    # Define an analytical solution u(x, y)
    u_exact_expr = 1 + x + 2 * y + sympy.sin(sympy.pi * x) * sympy.cos(sympy.pi * y)

    # Define the nonlinear coefficient k(u) = 1 + u^2
    k_expr = 1 + u_exact_expr**2

    # Compute the flux q = -k(u) * ∇u
    q_x = -k_expr * sympy.diff(u_exact_expr, x)
    q_y = -k_expr * sympy.diff(u_exact_expr, y)

    # Compute the source term f = ∇ ⋅ q = ∇ ⋅ (-k(u)∇u)
    f_expr = sympy.diff(q_x, x) + sympy.diff(q_y, y)

    # Simplify and convert to C code for FEniCSx Expressions
    u_code = sympy.printing.ccode(u_exact_expr)
    f_code = sympy.printing.ccode(f_expr)

    print(f"   ✓ Analytical u(x,y): {u_code}")
    print(f"   ✓ Derived source f(x,y): {f_code[:80]}...")  # Print a preview

    return u_code, f_code


def solve_nonlinear_poisson(u_code, f_code):
    """Solve the nonlinear Poisson equation -∇⋅((1+u²)∇u) = f"""

    print("\n--- Solving Nonlinear Poisson Problem ---")

    # 1. Create mesh and function space
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
    V = fem.functionspace(domain, ("Lagrange", 2))
    print("   ✓ Mesh and P2 function space created.")

    # 2. Define the exact solution as a dolfinx Function for BCs and error calculation
    u_exact = fem.Function(V)
    u_exact.interpolate(
        lambda x: eval(u_code, {"x": x, "sin": np.sin, "cos": np.cos, "M_PI": np.pi})
    )

    # 3. Define boundary conditions (Dirichlet, from exact solution)
    # First compute the required connectivity
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

    boundary_dofs = fem.locate_dofs_topological(
        V, domain.topology.dim - 1, mesh.exterior_facet_indices(domain.topology)
    )
    bc = fem.dirichletbc(u_exact, boundary_dofs)

    # 4. Define the nonlinear variational problem
    uh = fem.Function(V)  # The unknown trial function for the nonlinear problem
    v = ufl.TestFunction(V)

    # Define the nonlinear coefficient k(u) and source term f
    k = 1 + uh**2
    f = fem.Function(V)
    f.interpolate(
        lambda x: eval(
            f_code,
            {"x": x, "sin": np.sin, "cos": np.cos, "M_PI": np.pi, "pow": np.power},
        )
    )

    # Define the residual F(u; v) = 0
    # F = ∫ (k(u)∇u ⋅ ∇v - fv) dx
    F = ufl.inner(k * ufl.grad(uh), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx

    print("   ✓ Nonlinear variational problem defined.")

    # 5. Set up and solve the nonlinear problem using the correct DOLFINx API
    from dolfinx.nls.petsc import NewtonSolver

    # Create the nonlinear problem
    problem = fem_petsc.NonlinearProblem(F, uh, bcs=[bc])
    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    # Configure solver parameters
    solver.rtol = 1e-8
    solver.max_it = 20

    print("   Running Newton solver...")
    n, converged = solver.solve(uh)

    if converged:
        print(f"   ✓ SNES solver converged in {n} iterations.")
    else:
        print(f"   ✗ SNES solver did NOT converge after {n} iterations.")

    uh.name = "Numerical_Solution"
    return domain, V, uh, u_exact


def analyze_and_export(domain, V, uh, u_exact):
    """Compute error and export results"""
    print("\n--- Post-processing and Exporting ---")

    # 1. Compute the error in the L2 norm manually
    error_expr = (uh - u_exact) ** 2
    error_L2_squared = fem.assemble_scalar(fem.form(error_expr * ufl.dx))
    error_L2 = np.sqrt(error_L2_squared)

    # Compute max pointwise error
    error_max_dofs = uh.x.array - u_exact.x.array
    error_max = np.max(np.abs(error_max_dofs))

    print(f"   ✓ L2 Error: {error_L2:.4e}")
    print(f"   ✓ Max point-wise error: {error_max:.4e}")

    # 2. Export solution for visualization
    # Create P1 functions for XDMF export (compatibility with mesh degree)
    V_export = fem.functionspace(domain, ("Lagrange", 1))
    uh_export = fem.Function(V_export)
    uh_export.interpolate(uh)
    uh_export.name = "Numerical_Solution"

    u_exact_export = fem.Function(V_export)
    u_exact_export.interpolate(u_exact)
    u_exact_export.name = "Exact_Solution"

    error_export = fem.Function(V_export)
    error_export.x.array[:] = uh_export.x.array - u_exact_export.x.array
    error_export.name = "Error_Field"

    output_path = "output/nonlinear_poisson"
    with io.XDMFFile(domain.comm, f"{output_path}_solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh_export)
        xdmf.write_function(u_exact_export)

    with io.XDMFFile(domain.comm, f"{output_path}_error.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(error_export)

    print(f"   ✓ Exported solution and error fields to XDMF.")


def main():
    """Main function to run the nonlinear Poisson analysis"""

    print("=" * 60)
    print("NONLINEAR POISSON (p-LAPLACIAN TYPE) ANALYSIS")
    print("=" * 60)

    # 1. Generate symbolic expressions
    u_code, f_code = generate_analytical_solution()

    # 2. Solve the problem
    domain, V, uh, u_exact = solve_nonlinear_poisson(u_code, f_code)

    # 3. Analyze and export
    analyze_and_export(domain, V, uh, u_exact)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("To visualize, open '../../output/nonlinear_poisson_solution.xdmf'")
    print("and '../../output/nonlinear_poisson_error.xdmf' in ParaView.")
    print("=" * 60)


if __name__ == "__main__":
    main()
