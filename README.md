# FEniCSx Learning Project

## Overview
This repository contains examples, utilities, and detailed documentation for setting up and working with the FEniCSx finite element software on a Windows machine. The primary focus is on **FEniCSx v0.9.0**, using the recommended **Windows Subsystem for Linux (WSL2)** approach.

The goal is to provide a reliable path from a fresh Windows installation to solving and visualizing partial differential equations (PDEs).

## Directory Structure

```
.
├── README.md              # This file: your main guide
├── docs/
│   └── lessons_learned.md  # Detailed technical notes and discoveries
├── examples/
│   ├── 01_poisson_basic/   # Solves a simple linear elliptic PDE (Poisson equation).
│   ├── 02_heat_conduction_steady/ # Solves steady-state heat transfer with mixed BCs.
│   ├── 03_thermal_stress_analysis/ # Coupled thermo-mechanical simulation.
│   ├── 04_nonlinear_poisson/ # Solves a nonlinear elliptic PDE with a Newton solver.
│   ├── 05_transient_heat_equation/ # Solves the time-dependent heat (diffusion) equation.
│   └── 06_navier_stokes_laminar_flow/ # Solves the incompressible Navier-Stokes equations for CFD.
├── helpers/                # Reusable utility and diagnostic scripts
│   ├── check_file_health.py # Comprehensive diagnostic tool for output files.
│   ├── open_xdmf.py        # Helper for opening files in VisIt CLI.
│   └── pv_quicklook_gui.py # Lightweight ParaView GUI for quick visualization.
└── output/                 # For all generated files (.vtk, .xdmf, .png, etc.)
    └── .gitkeep            # Ensures the directory is tracked by Git.
```

---


## Installation and Setup on Windows (Step-by-Step)

FEniCSx does not run natively on Windows. The recommended and most robust method is to use the Windows Subsystem for Linux (WSL2).

### Step 1: Install WSL2 and Ubuntu
First, enable WSL2 on your Windows machine and install a Linux distribution.

1.  **Open PowerShell as Administrator.**
2.  **Install WSL2 and Ubuntu:** This single command handles everything.
    ```powershell
    wsl --install
    ```
3.  **Restart your computer** when prompted. After restarting, Ubuntu will complete its installation. You will be asked to create a username and password for your new Linux environment.

### Step 2: Set Up the FEniCSx Environment inside WSL2
Now, open your new Ubuntu terminal (you can find it in the Start Menu). All the following commands are run inside this terminal. You will need a package manager to create a dedicated Python environment; we recommend `mamba` for its speed, but standard `conda` also works perfectly.

**Option A: Install Miniconda (to use `conda`)**
This is the standard approach if you prefer to use `conda` directly.

1.  **Download the Miniconda Installer:**
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    ```
2.  **Run the Installer Script:**
    ```bash
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
    Follow the on-screen prompts. It is recommended to accept the default location and agree to run `conda init` when asked. This will set up your shell to use `conda`.
3.  **Restart your Terminal:** Close and reopen your Ubuntu terminal for the changes to take effect. Your command prompt should now start with `(base)`.

**Option B: Install Mambaforge (to use `mamba`, recommended for speed)**
Mamba is a parallel, drop-in replacement for `conda` that significantly speeds up package installation.

1.  **Download the Mambaforge Installer:**
    ```bash
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    ```
2.  **Run the Installer Script (non-interactive):**
    ```bash
    bash Mambaforge-*.sh -b -p "${HOME}/mambaforge"
    ```
3.  **Activate and Initialize Mamba:**
    ```bash
    source "${HOME}/mambaforge/bin/activate"
    mamba init
    ```
4.  **Restart your Terminal:** Close and reopen your Ubuntu terminal. Your command prompt should now start with `(base)`.

### Step 3: Create and Activate the FEniCSx Environment
This step is the same whether you installed Miniconda or Mambaforge.

1.  **Create the environment with all necessary packages.** Use the command corresponding to your installer:

    *   **If you chose Mamba:**
        ```bash
        mamba create -n fenicsx-env -c conda-forge fenics-dolfinx mpich pyvista h5py matplotlib sympy tqdm gmsh
        ```
    *   **If you chose Conda:** (This will be slower)
        ```bash
        conda create -n fenicsx-env
        conda activate fenicsx-env
        conda install -c conda-forge fenics-dolfinx mpich pyvista h5py matplotlib sympy tqdm gmsh
        ```

2.  **Activate the new environment:**
    ```bash
    conda activate fenicsx-env
    ```
    Your terminal prompt should now change to `(fenicsx-env)`.

### Step 4: Clone this Repository
Clone this project to get all the examples and helper scripts.

```bash
# Navigate to your home directory and clone the project
cd ~
git clone https://Foadsf/FEniCSx-Learning-Project.git
cd FEniCSx-Learning-Project
```

### Step 5: Install ParaView on Windows
For high-quality 3D visualization, we will use ParaView on the Windows host.

1.  Download and install ParaView from the [official website](https://www.paraview.org/download/).
2.  The typical installation path is `C:\Program Files\ParaView X.Y.Z`.

You are now fully set up!

---

## Running the Poisson Example

This workflow demonstrates how to solve a PDE inside WSL2 and visualize the results using tools on both Linux and Windows.

**1. Run the Solver:**
   Navigate to the example directory and run the Python script.
   ```bash
   # Make sure you are in the (fenicsx-env) environment
   cd examples/01_poisson_basic/
   python3 poisson_demo.py
   ```
   This will solve the PDE and create `poisson_solution.xdmf` and `.h5` files in the `output` directory.

**2. (Optional) Export to More Formats:**
   The default XDMF format can sometimes be problematic for ParaView. The included helper script exports the solution to multiple, more robust formats like VTK.
   ```bash
   # Navigate to the helpers directory
   cd ../../helpers/
   python3 poisson_vtk_export.py
   ```

**3. Visualize with Python (Quick Check):**
   For a quick plot without leaving the terminal environment:
   ```bash
   # Still in the helpers directory
   python3 simple_visualization.py
   ```
   This will generate `poisson_visualization.png` in the `output` directory and may open an interactive Matplotlib window.

**4. Visualize with ParaView (High Quality):**
   The most powerful way to inspect the 3D solution is with ParaView. Run this command from your WSL2 terminal:
   ```bash
   # The path must point to your Windows ParaView installation
   "/mnt/c/Program Files/ParaView 5.13.1/bin/paraview.exe" ../output/poisson_simple.vtk
   ```
   This command launches the Windows ParaView application and automatically loads the solution file from your WSL2 filesystem.

---

## Advanced Examples

### Heat Conduction with Mixed Boundary Conditions
A more realistic engineering problem demonstrating advanced boundary conditions:

```bash
cd examples/02_heat_conduction_steady/
python3 02_heat_conduction_steady.py
```

This example shows:
- **Mixed boundary conditions**: Dirichlet (fixed temperature), Robin (convective cooling), and Neumann (insulated)
- **Real material properties**: Aluminum thermal conductivity and heat capacity
- **Engineering analysis**: Heat flux calculations and thermal resistance
- **Multiple export formats**: VTK, XDMF, and raw data for maximum compatibility

### Thermal Stress Analysis (Coupled Physics)
A coupled thermo-mechanical analysis combining heat transfer and structural mechanics:

```bash
cd examples/03_thermal_stress_analysis/
python3 thermal_stress_analysis.py
```

Features:
- **Coupled physics**: Temperature field drives thermal expansion and stress
- **Realistic materials**: Steel properties with thermal expansion coefficients
- **Engineering outputs**: von Mises stress, displacement fields, and strain analysis
- **Multi-format export**: Handles complex multi-field problems

### Nonlinear PDE Solving
A comprehensive example of nonlinear partial differential equations using manufactured solutions:

```bash
cd examples/04_nonlinear_poisson/
python3 nonlinear_poisson_p_laplacian.py
```

This advanced example demonstrates:
- **Nonlinear PDEs**: p-Laplacian type equation with solution-dependent coefficients
- **Manufactured solutions**: Using SymPy for symbolic computation of exact solutions
- **Newton solver**: Iterative solution of nonlinear systems
- **Error analysis**: L2 and pointwise error computation against analytical solutions
- **High-order elements**: P2 finite elements with P1 export for visualization

The solver achieves machine precision accuracy (errors ~10^-5) for complex nonlinear problems.

### Time-Dependent PDEs
A comprehensive example of transient heat conduction with time evolution:

```bash
cd examples/05_transient_heat_equation/
python3 transient_heat_conduction.py
```

This time-dependent example features:
- **Transient heat equation**: ∂u/∂t = α∇²u with thermal diffusivity
- **Time discretization**: Backward Euler scheme for stability
- **Complex boundary conditions**: Central heat source with fixed outer boundaries
- **Time-stepping simulation**: 200 time steps over 10 seconds
- **Animation output**: Time series data for ParaView visualization
- **Progress tracking**: Real-time simulation progress display

The simulation models a copper plate with a central heat source, showing realistic heat diffusion patterns over time.

### Computational Fluid Dynamics (CFD)
A complete implementation of the incompressible Navier-Stokes equations for laminar flow:

```bash
cd examples/06_navier_stokes_laminar_flow/
python3 karman_vortex_street.py
```

This advanced CFD example demonstrates:
- **Incompressible Navier-Stokes equations**: Full momentum and continuity equations
- **IPCS fractional step method**: Incremental Pressure Correction Scheme for efficiency
- **Complex geometry**: GMSH integration for channel with cylindrical obstacle
- **Taylor-Hood elements**: P2 velocity, P1 pressure (inf-sup stable)
- **Realistic boundary conditions**: Parabolic inflow, no-slip walls, pressure outflow
- **Vortex shedding**: Kármán vortex street formation behind cylinder
- **Long-time simulation**: 8000 time steps showing flow development

The simulation captures the classic fluid dynamics phenomenon of periodic vortex shedding.

## Diagnostic and Visualization Tools

### File Health Checker
A comprehensive diagnostic tool for troubleshooting output files:

```bash
# Check individual files
python3 helpers/check_file_health.py output/solution.vtk
python3 helpers/check_file_health.py output/thermal_stress.bp

# Check entire output directory
python3 helpers/check_file_health.py output/
```

The health checker:
- **Validates file formats**: VTK, XDMF, HDF5, PVD, and ADIOS2 BP files
- **Diagnoses issues**: Identifies corruption, missing references, and format problems
- **Visualizes healthy files**: Automatically plots data from valid files
- **Provides recommendations**: Suggests fixes for common problems

### Visualization Pipeline
Multiple visualization options for different needs:

1. **Simple VTK files** (most compatible with ParaView)
2. **Python matplotlib** (built-in plots and analysis)
3. **ADIOS2 BP format** (for large-scale simulations)
4. **Raw data export** (CSV, NumPy arrays for custom analysis)


---

## FEniCSx v0.9.0 API Notes

This project uses FEniCSx v0.9.0. A key lesson learned is that this version has a specific API for PETSc-based solvers.

-   **Always import the PETSc submodule:** `from dolfinx.fem import petsc as fem_petsc`
-   **Use `fem_petsc` for assembly:** `fem_petsc.assemble_matrix()` returns a `PETSc.Mat` object.
-   **Use `A.assemble()`:** PETSc matrices require a final assembly call after creation.
-   **Solver interface:** Pass the PETSc vector from the solution function: `solver.solve(b, uh.x.petsc_vec)`.

For a deep dive into the API discoveries and debugging process, see `docs/lessons_learned.md`.

## Troubleshooting

-   **`AttributeError: 'MatrixCSR' object has no attribute 'assemble'`**
    -   **Cause:** You are using the base `fem.assemble_matrix()` which returns a native C++ object.
    -   **Solution:** Use `fem_petsc.assemble_matrix()` which returns a PETSc-compatible matrix.
-   **ParaView crashes when opening `.xdmf` file**
    -   **Cause:** Compatibility issue between HDF5 library versions.
    -   **Solution:** Use the `poisson_vtk_export.py` helper to generate a simpler `.vtk` file, which is much more reliable.
-   **`ModuleNotFoundError`**
    -   **Solution:** Make sure your `fenicsx-env` conda environment is activated. Install any missing packages with `mamba install <package-name>`.

---

### PV QuickLook (ParaView)

A lightweight, cross-platform GUI to quickly preview simulation results without using the full ParaView UI.

**Script:** `helpers/pv_quicklook_gui.py`
**Supported inputs:** `.xdmf/.xmf`, `.pvd`, `.vtu/.pvtu`, `.vtk`, `.bp` (if available), and `.h5/.hdf5` via a companion `.xdmf`.

#### Quick start (Windows example)

```bash
python helpers\pv_quicklook_gui.py -v output\transient_heat_solution.xdmf
````

* In the GUI, set **ParaView GUI (optional)** to your ParaView desktop binary.
  Default path on Windows:
  `C:\Program Files\ParaView 5.13.1\bin\paraview.exe`
* The app will **infer `pvpython.exe`** from that path and render **off-process** using ParaView’s Python.
  (This avoids Python/ABI mismatches; your local Python can be 3.13+, while ParaView’s is 3.10.)
* Use the simple controls (representation, color by, time step, camera) and **Save Screenshot…**.
* **Open in ParaView GUI** launches the full ParaView if you want to deep-dive.

#### CLI options

* `-v, --verbose` – prints a diagnostic report (Python, PATH, sys.path, dataset, etc.) before the GUI starts.

#### Notes & troubleshooting

* If the preview fails, the **status bar** shows the exact error captured from `pvpython` (e.g., missing plugin, invalid XDMF, etc.).
* Rendering is done offscreen; no desktop OpenGL context is required.
* For `.h5` data, prefer providing an `.xdmf` sidecar describing topology/geometry.

---

*Generated: September 8, 2025*
*FEniCSx Version: 0.9.0*
*Environment: WSL2 Ubuntu + Windows ParaView*
