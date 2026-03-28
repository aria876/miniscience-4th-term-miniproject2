#!/usr/bin/env python3
# visualize_solution.py
# Simple Python visualization of the Poisson solution
import numpy as np
import matplotlib.pyplot as plt
from dolfinx import fem, mesh
from mpi4py import MPI
import pyvista as pv

def visualize_with_matplotlib():
    """Visualize using matplotlib (basic 2D plot)"""
    print("Creating visualization with matplotlib...")
    
    # Recreate the same mesh and function space
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Create the analytical solution for comparison
    u_exact = fem.Function(V)
    u_exact.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
    
    # Get coordinates and values
    coords = V.tabulate_dof_coordinates()
    values = u_exact.x.array
    
    # Create a simple scatter plot
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Scatter plot of solution values
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=values, cmap='viridis', s=50)
    plt.colorbar(scatter, label='Solution value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Poisson Solution (Analytical)')
    plt.axis('equal')
    
    # Plot 2: 3D surface plot
    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], values, c=values, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')
    ax.set_title('3D View')
    
    plt.tight_layout()
    plt.savefig('poisson_solution_matplotlib.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved as: poisson_solution_matplotlib.png")

def visualize_with_pyvista():
    """Visualize using PyVista (advanced 3D visualization)"""
    try:
        print("Creating advanced visualization with PyVista...")
        
        # Read the XDMF file directly with PyVista
        reader = pv.XdmfReader("poisson_solution.xdmf")
        mesh_pv = reader.read()
        
        # Create interactive plot
        plotter = pv.Plotter(shape=(1, 2), window_size=(1200, 600))
        
        # Plot 1: 2D colormap view
        plotter.subplot(0, 0)
        plotter.add_mesh(mesh_pv, scalars=mesh_pv.active_scalars_name, 
                        cmap='viridis', show_edges=True)
        plotter.view_xy()
        plotter.add_title("2D Colormap View")
        
        # Plot 2: 3D surface view
        plotter.subplot(0, 1)
        warped = mesh_pv.warp_by_scalar(factor=0.3)
        plotter.add_mesh(warped, scalars=mesh_pv.active_scalars_name, 
                        cmap='viridis', show_edges=True)
        plotter.add_title("3D Warped Surface")
        
        # Save screenshot
        plotter.screenshot('poisson_solution_pyvista.png')
        plotter.show()
        print("✓ Saved as: poisson_solution_pyvista.png")
        
    except ImportError:
        print("PyVista not installed. Install with: pip install pyvista")
    except Exception as e:
        print(f"PyVista visualization failed: {e}")

def create_simple_contour():
    """Create a simple contour plot"""
    print("Creating contour plot...")
    
    # Create a grid for contour plotting
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z = 1 + X**2 + 2*Y**2  # Analytical solution
    
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, label='u(x,y)')
    plt.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Poisson Solution - Contour Plot\nu = 1 + x² + 2y²')
    plt.axis('equal')
    plt.savefig('poisson_contour.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved as: poisson_contour.png")

if __name__ == "__main__":
    print("=== POISSON SOLUTION VISUALIZATION ===")
    
    # Check if solution file exists
    import os
    if not os.path.exists("poisson_solution.xdmf"):
        print("❌ poisson_solution.xdmf not found. Run the Poisson solver first.")
        exit(1)
    
    print("Choose visualization method:")
    print("1. Matplotlib (basic, always works)")
    print("2. PyVista (advanced, requires installation)")
    print("3. Simple contour plot")
    print("4. All methods")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice in ['1', '4']:
        visualize_with_matplotlib()
    
    if choice in ['2', '4']:
        visualize_with_pyvista()
        
    if choice in ['3', '4']:
        create_simple_contour()
    
    print("\n✓ Visualization complete!")
