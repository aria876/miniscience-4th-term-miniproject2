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
    """Create a mesh for an elliptical domain using GMSH"""
    print("1. Creating mesh with GMSH...")

    gmsh.initialize()
    if comm.rank == 0:
        # Параметры эллипса
        center_x = 0.0
        center_y = 0.0
        center_z = 0.0
        radius_x = 5.0  # полуось по X
        radius_y = 3.0  # полуось по Y

        # ПАРАМЕТРЫ СЕТКИ
        mesh_size = 0.1  # <- ИЗМЕНЯЙТЕ ЭТО ЗНАЧЕНИЕ ДЛЯ КОНТРОЛЯ МЕЛКОСТИ
        # Чем меньше значение, тем мельче сетка
        # Для эллипса с радиусами 5 и 3:
        # mesh_size = 1.0 - грубая сетка
        # mesh_size = 0.5 - средняя сетка  
        # mesh_size = 0.2 - мелкая сетка
        # mesh_size = 0.1 - очень мелкая сетка
        
        # Устанавливаем глобальный размер сетки
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)



        
        # Маркеры для границ
        inlet_marker, outlet_marker, wall_marker, obstacle_marker = 1, 2, 3, 4
        
        # Создаем эллипс
        # Сначала создаем круг радиусом 1
        disk = gmsh.model.occ.addDisk(center_x, center_y, center_z, 1.0, 1.0)
        
        # Масштабируем круг до эллипса
        # Параметры: (объект, x_scale, y_scale, z_scale, center_x, center_y, center_z)
#        ellipse = gmsh.model.occ.dilate([(2, disk)], center_x, center_y, center_z, radius_x, radius_y, 1.0)
        
        gmsh.model.occ.synchronize()
        
        # Получаем все граничные кривые (в 2D это кривые, ограничивающие поверхность)
        all_curves = gmsh.model.getEntities(1)
        
        # Для эллипса все границы - это его контур
        # Нужно разбить их на части для разных физических границ
        # В зависимости от положения на эллипсе
        
        inlet_curves, outlet_curves, wall_curves = [], [], []
        
        # Классифицируем кривые по их геометрическому положению на эллипсе
        for curve in all_curves:
            # Получаем bounding box кривой
            bbox = gmsh.model.getBoundingBox(curve[0], curve[1])
            xmin, ymin, zmin, xmax, ymax, zmax = bbox
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            
            # Классификация на основе угла или координат
            # Например, можно разделить эллипс на 4 части:
            # - левая половина (x < 0) - вход
            # - правая половина (x > 0) - выход
            # - верхняя и нижняя части - стенки
            
            if x_center < -radius_x * 0.5:  # Левая часть эллипса
                inlet_curves.append(curve[1])
            elif x_center > radius_x * 0.5:  # Правая часть эллипса
                outlet_curves.append(curve[1])
            else:  # Верхняя и нижняя части (стенки)
                wall_curves.append(curve[1])
        
        # Применяем физические группы для маркировки границ
        # Вход
        if inlet_curves:
            inlet_group = gmsh.model.addPhysicalGroup(1, inlet_curves)
            gmsh.model.setPhysicalName(1, inlet_group, "inlet")
            gmsh.model.addPhysicalGroup(1, inlet_curves, inlet_marker)
        
        # Выход
        if outlet_curves:
            outlet_group = gmsh.model.addPhysicalGroup(1, outlet_curves)
            gmsh.model.setPhysicalName(1, outlet_group, "outlet")
            gmsh.model.addPhysicalGroup(1, outlet_curves, outlet_marker)
        
        # Стенки
        if wall_curves:
            wall_group = gmsh.model.addPhysicalGroup(1, wall_curves)
            gmsh.model.setPhysicalName(1, wall_group, "walls")
            gmsh.model.addPhysicalGroup(1, wall_curves, wall_marker)
        
        # Маркируем объемную область (2D поверхность эллипса)
        all_surfaces = gmsh.model.getEntities(2)
        fluid_surfaces = [s[1] for s in all_surfaces]
        fluid_group = gmsh.model.addPhysicalGroup(2, fluid_surfaces)
        gmsh.model.setPhysicalName(2, fluid_group, "fluid")
        
        # Генерируем сетку
        gmsh.model.mesh.generate(2)
        
        # Сохраняем mesh
        gmsh.write("mesh.msh")
        
    
    # Здесь нужно загрузить mesh в FEniCSx
    # from dolfinx.io import gmshio
    # mesh, subdomains, boundaries = gmshio.read_from_msh("mesh.msh", comm, rank=0)
    # return mesh, boundaries
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
        "T_final": 50.0,  # Final time [s]
        "num_steps": 500,  # Number of time steps
        # Material properties (Copper)
        "k": 401.0,  # Thermal conductivity [W/(m·K)]
        "rho": 8960.0,  # Density [kg/m³]
        "cp": 385.0,  # Specific heat [J/(kg·K)]
        # Initial and boundary conditions
        "T_initial": 0.0,  # Initial temperature of the plate [°C]
        "T_boundary": 0.0,  # Temperature at the outer boundary [°C]
        "T_source": 1000000.0,  # Temperature of the central heat source [°C]
    }
    # Calculate thermal diffusivity
    params["alpha"] = 10 * params["k"] / (params["rho"] * params["cp"])
    #Для волнового уравнения вместо \alpha будет c - скорость света в среде
    params["c"] = 1#np.sqrt(params["alpha"])
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

#    U1 = (coords[:,1] < 0.5)
#    U2 = (coords[:,0]**2 + (-2.24 + coords[:,1])**2 < 4)
#    inside = (coords[:, 1] < 0.9 * coords[:, 0])
#    inside = np.logical_and(U1, U2)
    #показатель преломления
#    n = 3
#    wave_speed.x.array[inside] = params["c"] / n
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
    T_generator = 2 #period
    w = 2 * np.pi / T_generator
    # Получаем координаты всех узлов (степеней свободы)
    coords = V.tabulate_dof_coordinates()  # shape = (num_dofs, 3) (x, y, z)
    # Определяем, какие узлы лежат внутри кругов
#   circle1 = (coords[:, 0]**2 + (coords[:, 1] - 0.25)**2) < 0.01   # радиус
#   circle2 = (coords[:, 0]**2 + (coords[:, 1] + 0.25)**2) < 0.01
    K1 = (coords[:, 0])**2 + (coords[:, 1])**2 < 0.01
    inside_generator = K1 #np.logical_or(circle1, circle2)

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
