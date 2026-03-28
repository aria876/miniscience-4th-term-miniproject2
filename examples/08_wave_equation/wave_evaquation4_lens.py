#!/usr/bin/env python3
# transient_heat_conduction.py
# Solves the time-dependent heat equation (diffusion equation).
#
# Problem Description:
# A rectangular plate is initially at a uniform room temperature. A circular
# region in the center is suddenly heated and maintained at a high temperature,
# acting as a constant heat source. The outer edges of the plate are held
# at room temperature. We simulate how the temperature field evolves over time.
#
# PDE: ∂u/∂t = α ∇²u + f
#  - u: Temperature
#  - t: Time
#  - α: Thermal diffusivity (alpha = k / (rho * cp))
#  - f: Heat source/sink term (zero in this case)

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
        L, H = 2, 2
        c_x, c_y, r = 0.2, 0.2, 0.05

        # Define geometry
        channel = gmsh.model.occ.addRectangle(-1, -1, 0, L, H)
#        cylinder = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

        # Cut the cylinder from the channel
        #i don't want to cut cylinder
        fluid_domain = channel
        gmsh.model.occ.synchronize()

        # Mark boundaries using geometric criteria instead of boundary iteration
#        inlet_marker, outlet_marker, wall_marker, obstacle_marker = 1, 2, 3, 4

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

#        # Add physical groups only if curves exist
#        if inlet_curves:
#            gmsh.model.addPhysicalGroup(1, inlet_curves, inlet_marker)
#        if outlet_curves:
#            gmsh.model.addPhysicalGroup(1, outlet_curves, outlet_marker)
#        if wall_curves:
#            gmsh.model.addPhysicalGroup(1, wall_curves, wall_marker)
#        if obstacle_curves:
#            gmsh.model.addPhysicalGroup(1, obstacle_curves, obstacle_marker)

        # Add physical group for the fluid domain
        gmsh.model.addPhysicalGroup(2, [fluid_domain], 5)  # без дырки
#        gmsh.model.addPhysicalGroup(2, [fluid_domain[0][1]], 5)

        # Set mesh resolution, мелкость сетки
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.009) #0.02)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01) #0.1)

        # Generate mesh
        gmsh.model.mesh.generate(2)

    # Convert to DOLFINx mesh
    domain, cell_markers, facet_markers = io.gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=2
    )

    gmsh.finalize()
    print(f"   ✓ Mesh created with {domain.topology.index_map(2).size_global} cells.")
    return domain










def create_problem_geometry():
    """Create a rectangular domain for the plate"""
    print("1. Creating problem geometry...")
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        points=[(-1.0, -1.0), (1.0, 1.0)],
        n=[50, 50],
        cell_type=mesh.CellType.triangle,
        diagonal=mesh.DiagonalType.crossed  # или diagonal="crossed"
    )
    print(f"   ✓ Mesh created with {domain.topology.index_map(2).size_global} cells.")
    return domain


def define_physical_parameters():
    """Define simulation time and material properties for copper"""
    params = {
        # Simulation parameters
        "T_final": 100.0,  # Final time [s]
        "num_steps": 500,  # Number of time steps
        # Material properties (Copper)
        "k": 401.0,  # Thermal conductivity [W/(m·K)]
        "rho": 8960.0,  # Density [kg/m³]
        "cp": 385.0,  # Specific heat [J/(kg·K)]
        # Initial and boundary conditions
        "T_initial": 25.0,  # Initial temperature of the plate [°C]
        "T_boundary": 25.0,  # Temperature at the outer boundary [°C]
        "T_source": 1000000.0,  # Temperature of the central heat source [°C]
    }
    # Calculate thermal diffusivity
    params["alpha"] = 10 * params["k"] / (params["rho"] * params["cp"])
    #Для волнового уравнения вместо \alpha будет c - скорость света в среде
    params["c"] = np.sqrt(params["alpha"])
    # Calculate time step size
    params["dt"] = params["T_final"] / params["num_steps"]

    print("2. Defining physical parameters...")
    print(f"   ✓ Material: Copper (α = {params['alpha']:.2e} m²/s)")
    print(
        f"   ✓ Simulation time: {params['T_final']} s, Time step (dt): {params['dt']:.3f} s"
    )
    return params


def setup_time_dependent_problem(domain, params):
    """Setup the variational formulation for the transient heat equation"""
    print("3. Setting up transient variational problem...")

    # Create function space
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Define trial and test functions
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Define function to hold the solution from the previous time step
    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.x.array[:] = params["T_initial"]

    # Пред предыдущее состояние температуры(trial function)
    u_n_n = fem.Function(V)
    u_n_n.name = "u_n_n"
    u_n_n.x.array[:] = params["T_initial"]

# Скорость волны
# {
    wave_speed = fem.Function(V)
    wave_speed.name = "wave_speed"
    #геометрия сред с различными показателями преломления
    coords = V.tabulate_dof_coordinates()
    wave_speed.x.array[:] = params["c"]

    U1 = (coords[:,1] < 0.5)
    U2 = (coords[:,0]**2 + (-2.24 + coords[:,1])**2 < 4)
#    inside = (coords[:, 1] < 0.9 * coords[:, 0])
    inside = np.logical_and(U1, U2)
    #показатель преломления
    n = 3
    wave_speed.x.array[inside] = params["c"] / n
# }


# {
    #Для начального возмущения
    # Получаем координаты всех узлов (степеней свободы)
#    coords = V.tabulate_dof_coordinates()  # shape = (num_dofs, 3) (x, y, z)

    # Определяем, какие узлы лежат внутри кругов
#    circle1 = (coords[:, 0]**2 + (coords[:, 1] - 0.5)**2) < 0.001   # радиус 0.001
#    circle2 = (coords[:, 0]**2 + (coords[:, 1] + 0.5)**2) < 0.001
#    inside = np.logical_or(circle1, circle2)
#    upper_bound = ((coords[:, 1]**2) < 0.1)
#    weight_bound = (((coords[:, 0] - 0.5)**2) < 0.01)
#    inside = np.logical_and(upper_bound, weight_bound)

    # Устанавливаем температуру источника в этих узлах
#    u_n.x.array[inside] = params["T_source"]
#    u_n_n.x.array[inside] = params["T_source"]
# }


    # Define constants
    alpha = fem.Constant(domain, default_scalar_type(params["alpha"]))
    dt = fem.Constant(domain, default_scalar_type(params["dt"]))

    # Weak form using Backward Euler time discretization:
    # (u - u_n)/dt = α ∇²u
    # ∫(u - u_n)v dx = ∫ α dt ∇²u v dx
    # ∫uv dx - ∫u_n v dx = -∫ α dt ∇u ⋅ ∇v dx + ∫ α dt (∇u ⋅ n)v ds
    # Rearranging for LHS (unknown u) and RHS (known u_n):
    # ∫(u + α dt ∇u ⋅ ∇v)v dx = ∫u_n v dx
    F = (
        u * v * ufl.dx
        + (wave_speed * wave_speed) * dt *dt * (ufl.dot(ufl.grad(u), ufl.grad(v))) * ufl.dx
        - 2.0 * u_n * v * ufl.dx + u_n_n * v * ufl.dx
    )

    a = fem.form(ufl.lhs(F))
    L = fem.form(ufl.rhs(F))

    print("   ✓ Variational forms (a, L) created for Backward Euler.")
    return V, a, L, u_n, u_n_n






def define_boundary_conditions(domain, V, params):
    """Define time-dependent boundary conditions"""

    # 1. Outer boundary condition (fixed at T_boundary)
    # Since all exterior facets are at the same temperature, we can do this simply.
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_dofs = fem.locate_dofs_topological(
        V, domain.topology.dim - 1, mesh.exterior_facet_indices(domain.topology)
    )
    bc_outer = fem.dirichletbc(
        default_scalar_type(params["T_boundary"]), boundary_dofs, V
    )

    # 2. Inner circular heat source (fixed at T_source)
    def source_region(x):
        # Определяем два круга с центрами (0, 0.5) и (0, -0.5) и радиусом 0.0001 (возможно, нужно увеличить)
        circle1 = (x[0]**2 + (x[1] - 0.5)**2) < 0.001
        circle2 = (x[0]**2 + (x[1] + 0.5)**2) < 0.001
        return np.logical_and(circle1, circle2)   # пустое значение для
#    источника

    source_dofs = fem.locate_dofs_geometrical(V, source_region)
    bc_source = fem.dirichletbc(default_scalar_type(params["T_source"]), source_dofs, V)

    print("4. Defining boundary conditions...")
    print(f"   ✓ Outer walls fixed at {params['T_boundary']} °C")
    print(f"   ✓ Central heat source fixed at {params['T_source']} °C")

    return [bc_outer, bc_source]




def run_transient_simulation(domain, V, a, L, u_n, u_n_n, bcs, params):
    """Run the time-stepping loop to solve the transient problem"""

    print("\n--- Running Transient Simulation ---")

    # 1. Assemble the time-independent parts of the system
    A = fem_petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()


    #2. Create solution function and output file
# { 
    uh = fem.Function(V)
    uh.name = "Temperature"

    output_path = "../../output/wave_solution.xdmf"
    xdmf = io.XDMFFile(domain.comm, output_path, "w")
    xdmf.write_mesh(domain)

    # Write initial condition (t=0)
    uh.x.array[:] = u_n.x.array
    xdmf.write_function(uh, 0.0)
# }


#Генератор гармонических волн
    T_generator = 10 #period
    w = 2 * np.pi / T_generator
    # Получаем координаты всех узлов (степеней свободы)
    coords = V.tabulate_dof_coordinates()  # shape = (num_dofs, 3) (x, y, z)
    # Определяем, какие узлы лежат внутри кругов
#   circle1 = (coords[:, 0]**2 + (coords[:, 1] - 0.25)**2) < 0.01   # радиус
#   circle2 = (coords[:, 0]**2 + (coords[:, 1] + 0.25)**2) < 0.01
    triangle = (coords[:, 1] > + 0.9)
    inside_generator = triangle #np.logical_or(circle1, circle2)

    # 3. Time-stepping loop
    t = 0.0
    progress = tqdm(range(params["num_steps"]), desc="Time-stepping")
    for i in progress:
        # Update time
        t += params["dt"]

        # Assemble the right-hand side vector (which depends on u_n)
        b = fem_petsc.assemble_vector(L)

        # Apply boundary conditions
        fem_petsc.apply_lifting(b, [a], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, bcs)

        # Setup solver
        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        # Solve for the current time step
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        # Update the solution for the next time step!!!!!!!!!
        u_n_n.x.array[:] = u_n.x.array
        u_n.x.array[:] = uh.x.array

        # Save solution to file at specified intervals
        if (i + 1) % 5 == 0:  # Save every 5 steps
            xdmf.write_function(uh, t)
            progress.set_postfix({"t": f"{t:.2f}s"})




# Генератор гармонических волн

        u_n.x.array[inside_generator] = 100 * np.cos(w * t) * params["T_source"]
        u_n_n.x.array[inside_generator] = 100 * np.cos(w * t) * params["T_source"]

    xdmf.close()
    print(f"   ✓ Simulation finished. Results saved to {output_path}")







def main():
    """Main function to run the transient heat analysis"""

    print("=" * 60)
    print("TRANSIENT HEAT CONDUCTION ANALYSIS")
    print("=" * 60)

    domain = create_mesh_with_obstacle(MPI.COMM_WORLD)
    params = define_physical_parameters()
    V, a, L, u_n, u_n_n = setup_time_dependent_problem(domain, params)
    bcs = define_boundary_conditions(domain, V, params)
    run_transient_simulation(domain, V, a, L, u_n, u_n_n, bcs, params)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("Use the time controls in ParaView to play the animation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
