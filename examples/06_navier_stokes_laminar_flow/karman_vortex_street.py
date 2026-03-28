#!/usr/bin/env python3
# karman_vortex_street.py
# Solves the transient, incompressible Navier-Stokes equations to simulate
# laminar flow around a cylinder, leading to a Kármán vortex street.
#
# Problem Description:
# A fluid enters a rectangular channel from the left with a parabolic velocity
# profile. It encounters a cylindrical obstacle, causing the flow to separate
# and form a periodic pattern of swirling vortices downstream. This is a classic
# benchmark problem in CFD known as "DFG 2D-2 benchmark".
#
# Equations: Incompressible Navier-Stokes
#   ∂u/∂t + (u ⋅ ∇)u = -∇p + ν∇²u + f   (Momentum)
#   ∇ ⋅ u = 0                          (Incompressibility)
#
#  - u: Fluid velocity (vector)
#  - p: Fluid pressure (scalar)
#  - ν: Kinematic viscosity (nu = mu / rho)
#  - f: Body force (zero in this case)
#
# Solution Method:
# We use the IPCS (Incremental Pressure Correction Scheme), a fractional-step
# method that decouples the velocity and pressure calculations for efficiency.
#  Step 1: Solve for a tentative velocity (u*) ignoring the pressure gradient.
#  Step 2: Solve a Poisson equation for the pressure correction (φ).
#  Step 3: Correct the velocity using the pressure correction gradient.

import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot, default_scalar_type
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm
import gmsh
import os


def create_mesh_with_obstacle(comm):
    """Create a mesh for a channel with a cylindrical obstacle using GMSH"""
    print("1. Creating mesh with GMSH...")

    gmsh.initialize()
    if comm.rank == 0:
        # Channel and obstacle parameters
        L, H = 2.2, 0.41
        c_x, c_y, r = 0.2, 0.2, 0.05

        # Define geometry
        channel = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        cylinder = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

        # Cut the cylinder from the channel
        fluid_domain, _ = gmsh.model.occ.cut([(2, channel)], [(2, cylinder)])
        gmsh.model.occ.synchronize()

        # Mark boundaries using geometric criteria instead of boundary iteration
        inlet_marker, outlet_marker, wall_marker, obstacle_marker = 1, 2, 3, 4

        # Get all boundaries
        all_surfaces = gmsh.model.getEntities(2)
        all_curves = gmsh.model.getEntities(1)

        inlet_curves, outlet_curves, wall_curves, obstacle_curves = [], [], [], []

        for curve in all_curves:
            # Get curve bounds to classify
            bbox = gmsh.model.getBoundingBox(curve[0], curve[1])
            xmin, ymin, zmin, xmax, ymax, zmax = bbox

            # Classify based on position
            if abs(xmin) < 1e-6 and abs(xmax) < 1e-6:  # x ≈ 0 (inlet)
                inlet_curves.append(curve[1])
            elif abs(xmin - L) < 1e-6 and abs(xmax - L) < 1e-6:  # x ≈ L (outlet)
                outlet_curves.append(curve[1])
            elif (abs(ymin) < 1e-6 and abs(ymax) < 1e-6) or (
                abs(ymin - H) < 1e-6 and abs(ymax - H) < 1e-6
            ):  # y ≈ 0 or y ≈ H (walls)
                wall_curves.append(curve[1])
            else:  # Internal curves (obstacle)
                obstacle_curves.append(curve[1])

        # Add physical groups only if curves exist
        if inlet_curves:
            gmsh.model.addPhysicalGroup(1, inlet_curves, inlet_marker)
        if outlet_curves:
            gmsh.model.addPhysicalGroup(1, outlet_curves, outlet_marker)
        if wall_curves:
            gmsh.model.addPhysicalGroup(1, wall_curves, wall_marker)
        if obstacle_curves:
            gmsh.model.addPhysicalGroup(1, obstacle_curves, obstacle_marker)

        # Add physical group for the fluid domain
        gmsh.model.addPhysicalGroup(2, [fluid_domain[0][1]], 5)

        # Set mesh resolution
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.02)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)

        # Generate mesh
        gmsh.model.mesh.generate(2)

    # Convert to DOLFINx mesh
    domain, cell_markers, facet_markers = io.gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=2
    )

    gmsh.finalize()
    print(f"   ✓ Mesh created with {domain.topology.index_map(2).size_global} cells.")
    return domain, facet_markers


def define_physical_parameters():
    """Define simulation time, fluid properties, and BC values"""
    params = {
        # Simulation control
        "T_final": 8.0,
        "dt": 0.0005,
        # Fluid properties (Glycerin-like)
        "rho": 1.26,  # kg/m^3
        "nu": 0.001,  # m^2/s (kinematic viscosity)
        # Boundary condition: Parabolic inflow U_avg = 1.0 m/s
        "U_max": 1.5,
    }
    params["num_steps"] = int(params["T_final"] / params["dt"])
    print("2. Defining physical and simulation parameters...")
    print(
        f"   ✓ Simulating for {params['T_final']}s with dt={params['dt']}s ({params['num_steps']} steps)"
    )
    return params


def setup_function_spaces(domain):
    """Setup mixed function space for velocity and pressure"""
    # Taylor-Hood elements: P2 for velocity, P1 for pressure
    V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", 1))
    return V, Q


def setup_boundary_conditions(domain, V, Q, ft, params):
    """Setup boundary conditions for inflow, outflow, and no-slip walls"""
    print("3. Setting up boundary conditions...")

    # Inflow velocity profile: u(y) = 4 * U_max * y * (H-y) / H^2
    H = 0.41
    U_max = params["U_max"]

    class InflowVelocity:
        def __init__(self, U_max, H):
            self.U_max = U_max
            self.H = H

        def __call__(self, x):
            return np.vstack(
                (
                    4 * self.U_max * x[1] * (self.H - x[1]) / (self.H**2),
                    np.zeros(x.shape[1]),
                )
            )

    inflow_profile = InflowVelocity(U_max, H)
    inlet_marker = 1
    inflow_dofs = fem.locate_dofs_topological(V, ft.dim, ft.find(inlet_marker))

    # Create a Function for the inflow profile
    u_inflow = fem.Function(V)
    u_inflow.interpolate(inflow_profile)

    bc_inflow = fem.dirichletbc(u_inflow, inflow_dofs)

    # No-slip BCs for walls and cylinder
    wall_marker, obstacle_marker = 3, 4
    u_zero = np.array([0, 0], dtype=default_scalar_type)
    wall_dofs = fem.locate_dofs_topological(V, ft.dim, ft.find(wall_marker))
    obstacle_dofs = fem.locate_dofs_topological(V, ft.dim, ft.find(obstacle_marker))
    bc_wall = fem.dirichletbc(fem.Constant(domain, u_zero), wall_dofs, V)
    bc_obstacle = fem.dirichletbc(fem.Constant(domain, u_zero), obstacle_dofs, V)

    bcu = [bc_inflow, bc_wall, bc_obstacle]

    # Outflow pressure BC (applied during pressure correction step)
    outlet_marker = 2
    p_zero = default_scalar_type(0)
    outlet_dofs = fem.locate_dofs_topological(Q, ft.dim, ft.find(outlet_marker))
    bcp = [fem.dirichletbc(p_zero, outlet_dofs, Q)]

    print("   ✓ Inflow, no-slip, and outflow BCs defined.")
    return bcu, bcp


def setup_ipcs_solver(domain, V, Q, params, bcu, bcp):
    """Setup the variational forms and solvers for the IPCS scheme"""
    print("4. Setting up IPCS solver...")

    # Trial and test functions
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    # Solution functions at current (k) and previous (k-1) time steps
    u_k, u_km1 = fem.Function(V), fem.Function(V)
    p_k, p_km1 = fem.Function(Q), fem.Function(Q)
    phi = fem.Function(Q)

    # Constants
    dt = fem.Constant(domain, default_scalar_type(params["dt"]))
    nu = fem.Constant(domain, default_scalar_type(params["nu"]))

    # --- Step 1: Tentative velocity (u*) ---
    u_mid = 0.5 * (u + u_km1)
    F1 = (
        (1 / dt) * ufl.inner(u - u_km1, v) * ufl.dx
        + ufl.inner(ufl.dot(u_km1, ufl.grad(u_mid)), v) * ufl.dx
        + nu * ufl.inner(ufl.grad(u_mid), ufl.grad(v)) * ufl.dx
        - ufl.inner(p_km1, ufl.div(v)) * ufl.dx
    )

    a1 = fem.form(ufl.lhs(F1))
    L1 = fem.form(ufl.rhs(F1))

    # --- Step 2: Pressure correction (φ) ---
    a2 = fem.form(ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx)
    L2 = fem.form(-(1 / dt) * ufl.div(u_k) * q * ufl.dx)

    # --- Step 3: Velocity correction ---
    a3 = fem.form(ufl.dot(u, v) * ufl.dx)
    L3 = fem.form(ufl.dot(u_k, v) * ufl.dx - dt * ufl.dot(ufl.grad(phi), v) * ufl.dx)

    # Pre-assemble matrices
    A1 = fem_petsc.assemble_matrix(a1, bcs=bcu)
    A2 = fem_petsc.assemble_matrix(a2, bcs=bcp)
    A3 = fem_petsc.assemble_matrix(a3)
    A1.assemble(), A2.assemble(), A3.assemble()

    # Create vectors
    b1, b2, b3 = (
        fem_petsc.create_vector(L1),
        fem_petsc.create_vector(L2),
        fem_petsc.create_vector(L3),
    )

    # Create KSP solvers
    solver1 = PETSc.KSP().create(domain.comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    solver1.getPC().setType(PETSc.PC.Type.JACOBI)

    solver2 = PETSc.KSP().create(domain.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.MINRES)
    solver2.getPC().setType(PETSc.PC.Type.HYPRE)

    solver3 = PETSc.KSP().create(domain.comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    solver3.getPC().setType(PETSc.PC.Type.JACOBI)

    print("   ✓ IPCS forms, matrices, and solvers configured.")
    return {
        "u_k": u_k,
        "u_km1": u_km1,
        "p_k": p_k,
        "p_km1": p_km1,
        "phi": phi,
        "a1": a1,
        "L1": L1,
        "A1": A1,
        "b1": b1,
        "solver1": solver1,
        "a2": a2,
        "L2": L2,
        "A2": A2,
        "b2": b2,
        "solver2": solver2,
        "a3": a3,
        "L3": L3,
        "A3": A3,
        "b3": b3,
        "solver3": solver3,
    }


def run_simulation(domain, solver_data, bcu, bcp, params):
    """Run the main time-stepping loop"""
    print("\n--- Running Transient Simulation ---")

    # Unpack solver data
    u_k, u_km1, p_k, p_km1, phi = (
        solver_data["u_k"],
        solver_data["u_km1"],
        solver_data["p_k"],
        solver_data["p_km1"],
        solver_data["phi"],
    )
    L1, b1, solver1 = solver_data["L1"], solver_data["b1"], solver_data["solver1"]
    L2, b2, solver2 = solver_data["L2"], solver_data["b2"], solver_data["solver2"]
    L3, b3, solver3 = solver_data["L3"], solver_data["b3"], solver_data["solver3"]

    # Output file and export function spaces
    output_path = "output/karman_vortex_street.xdmf"
    xdmf = io.XDMFFile(domain.comm, output_path, "w")
    xdmf.write_mesh(domain)

    # Create P1 function spaces for export compatibility
    V_export = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    Q_export = fem.functionspace(domain, ("Lagrange", 1))
    u_export = fem.Function(V_export)
    p_export = fem.Function(Q_export)
    u_export.name = "Velocity"
    p_export.name = "Pressure"

    t = 0.0
    progress = tqdm(range(params["num_steps"]), desc="Time-stepping")
    for i in progress:
        t += params["dt"]

        # --- Step 1: Solve for tentative velocity u* ---
        with b1.localForm() as loc_b:
            loc_b.set(0)
        fem_petsc.assemble_vector(b1, L1)
        fem_petsc.apply_lifting(b1, [solver_data["a1"]], bcs=[bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b1, bcu)
        solver1.solve(b1, u_k.x.petsc_vec)
        u_k.x.scatter_forward()

        # --- Step 2: Solve for pressure correction φ ---
        with b2.localForm() as loc_b:
            loc_b.set(0)
        fem_petsc.assemble_vector(b2, L2)
        fem_petsc.apply_lifting(b2, [solver_data["a2"]], bcs=[bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b2, bcp)
        solver2.solve(b2, phi.x.petsc_vec)
        phi.x.scatter_forward()

        # --- Step 3: Correct velocity ---
        with b3.localForm() as loc_b:
            loc_b.set(0)
        fem_petsc.assemble_vector(b3, L3)
        fem_petsc.set_bc(b3, bcu)  # No-slip is enforced on the corrected velocity
        solver3.solve(b3, u_k.x.petsc_vec)
        u_k.x.scatter_forward()

        # Update previous solutions
        u_km1.x.array[:] = u_k.x.array
        p_km1.x.array[:] = (
            p_k.x.array + phi.x.array
        )  # Update pressure for next tentative step

        # Save to file periodically
        if (i + 1) % 50 == 0:  # Save every 50 steps
            # Interpolate to P1 for export
            u_export.interpolate(u_k)
            p_export.interpolate(p_km1)
            xdmf.write_function(u_export, t)
            xdmf.write_function(p_export, t)
            progress.set_postfix({"t": f"{t:.2f}s"})

    xdmf.close()
    print(f"   ✓ Simulation finished. Results saved to {output_path}")


def main():
    """Main function to run the Navier-Stokes simulation"""
    print("=" * 60)
    print("LAMINAR FLOW AROUND A CYLINDER (KÁRMÁN VORTEX STREET)")
    print("=" * 60)

    domain, ft = create_mesh_with_obstacle(MPI.COMM_WORLD)
    params = define_physical_parameters()
    V, Q = setup_function_spaces(domain)
    bcu, bcp = setup_boundary_conditions(domain, V, Q, ft, params)
    solver_data = setup_ipcs_solver(domain, V, Q, params, bcu, bcp)
    run_simulation(domain, solver_data, bcu, bcp, params)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("To visualize, open '../../output/karman_vortex_street.xdmf' in ParaView.")
    print(
        "In ParaView, you can color by the 'u_k' field and use the 'Glyph' filter to show velocity vectors."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
