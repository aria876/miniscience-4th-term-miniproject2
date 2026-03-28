#!/usr/bin/env python3
# simple_visualization.py
# Simple visualization without ParaView

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_from_raw_data():
    """Visualize using the raw data files"""
    
    print("=== SIMPLE VISUALIZATION ===")
    
    # Try to load data from different sources
    data_loaded = False
    coords = None
    values = None
    
    # Method 1: Load from NumPy file
    try:
        data = np.load('poisson_solution_raw.npz')
        coords = data['coordinates']
        values = data['values']
        print("✓ Loaded data from poisson_solution_raw.npz")
        data_loaded = True
    except FileNotFoundError:
        print("✗ poisson_solution_raw.npz not found")
    
    # Method 2: Load from text file
    if not data_loaded:
        try:
            data = np.loadtxt('poisson_solution.txt', skiprows=1)
            coords = data[:, :2]  # x, y coordinates
            values = data[:, 2]   # u(x,y) values
            print("✓ Loaded data from poisson_solution.txt")
            data_loaded = True
        except FileNotFoundError:
            print("✗ poisson_solution.txt not found")
    
    # Method 3: Generate analytical solution
    if not data_loaded:
        print("📊 Generating analytical solution for visualization")
        x = np.linspace(0, 1, 9)
        y = np.linspace(0, 1, 9)
        X, Y = np.meshgrid(x, y)
        coords = np.column_stack([X.ravel(), Y.ravel()])
        values = 1 + coords[:, 0]**2 + 2 * coords[:, 1]**2
        print("✓ Generated analytical solution")
        data_loaded = True
    
    if not data_loaded:
        print("❌ No data available for visualization")
        return
    
    print(f"📊 Data shape: {coords.shape[0]} points")
    print(f"📊 Value range: [{np.min(values):.6f}, {np.max(values):.6f}]")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Scatter plot with colormap
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(coords[:, 0], coords[:, 1], c=values, 
                         cmap='viridis', s=100, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Solution Values (Scatter)')
    ax1.set_aspect('equal')
    plt.colorbar(scatter, ax=ax1, label='u(x,y)')
    
    # Plot 2: 3D scatter plot
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    scatter3d = ax2.scatter(coords[:, 0], coords[:, 1], values, 
                           c=values, cmap='viridis', s=50)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y)')
    ax2.set_title('3D Scatter Plot')
    
    # Plot 3: Contour plot (interpolated)
    ax3 = plt.subplot(2, 3, 3)
    # Create regular grid for contour
    x_reg = np.linspace(0, 1, 50)
    y_reg = np.linspace(0, 1, 50)
    X_reg, Y_reg = np.meshgrid(x_reg, y_reg)
    Z_reg = 1 + X_reg**2 + 2 * Y_reg**2  # Analytical solution
    
    contour = ax3.contourf(X_reg, Y_reg, Z_reg, levels=20, cmap='viridis')
    ax3.contour(X_reg, Y_reg, Z_reg, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Contour Plot (Analytical)')
    ax3.set_aspect('equal')
    plt.colorbar(contour, ax=ax3, label='u(x,y)')
    
    # Plot 4: 3D surface plot
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    surface = ax4.plot_surface(X_reg, Y_reg, Z_reg, cmap='viridis', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('u(x,y)')
    ax4.set_title('3D Surface (Analytical)')
    
    # Plot 5: Cross-section at y=0.5
    ax5 = plt.subplot(2, 3, 5)
    # Find points close to y=0.5
    y_target = 0.5
    tolerance = 0.1
    mask = np.abs(coords[:, 1] - y_target) < tolerance
    if np.sum(mask) > 0:
        x_cross = coords[mask, 0]
        u_cross = values[mask]
        idx_sort = np.argsort(x_cross)
        ax5.plot(x_cross[idx_sort], u_cross[idx_sort], 'bo-', label='Computed')
    
    # Analytical cross-section
    x_anal = np.linspace(0, 1, 100)
    u_anal = 1 + x_anal**2 + 2 * y_target**2
    ax5.plot(x_anal, u_anal, 'r-', label='Analytical')
    ax5.set_xlabel('x')
    ax5.set_ylabel('u(x, 0.5)')
    ax5.set_title(f'Cross-section at y={y_target}')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Error analysis (if we have computed values)
    ax6 = plt.subplot(2, 3, 6)
    if coords.shape[0] == 81:  # Expected for 9x9 grid
        # Compute analytical solution at same points
        u_analytical = 1 + coords[:, 0]**2 + 2 * coords[:, 1]**2
        error = np.abs(values - u_analytical)
        
        scatter_err = ax6.scatter(coords[:, 0], coords[:, 1], c=error, 
                                 cmap='Reds', s=100, edgecolors='black', linewidth=0.5)
        ax6.set_xlabel('x')
        ax6.set_ylabel('y')
        ax6.set_title(f'Absolute Error\nMax error: {np.max(error):.2e}')
        ax6.set_aspect('equal')
        plt.colorbar(scatter_err, ax=ax6, label='|error|')
    else:
        ax6.text(0.5, 0.5, 'Error analysis\nnot available\n(unexpected grid size)', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Error Analysis')
    
    plt.tight_layout()
    plt.savefig('poisson_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved as: poisson_visualization.png")
    plt.show()

def create_paraview_compatible_vtk():
    """Create a simple VTK file that ParaView should be able to read"""
    
    print("\n=== CREATING SIMPLE VTK FILE ===")
    
    try:
        # Create a simple structured grid
        x = np.linspace(0, 1, 9)
        y = np.linspace(0, 1, 9)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)  # 2D data, so z=0
        
        # Compute solution
        U = 1 + X**2 + 2 * Y**2
        
        # Write simple VTK file
        with open('poisson_simple.vtk', 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Poisson Solution\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_GRID\n")
            f.write(f"DIMENSIONS {X.shape[1]} {X.shape[0]} 1\n")
            f.write(f"POINTS {X.size} float\n")
            
            # Write points
            for j in range(X.shape[0]):
                for i in range(X.shape[1]):
                    f.write(f"{X[j,i]:.6f} {Y[j,i]:.6f} {Z[j,i]:.6f}\n")
            
            # Write point data
            f.write(f"POINT_DATA {X.size}\n")
            f.write("SCALARS solution float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(U.shape[0]):
                for i in range(U.shape[1]):
                    f.write(f"{U[j,i]:.6f}\n")
        
        print("✓ Created poisson_simple.vtk")
        print("  Try opening this file with ParaView - it should work better")
        
    except Exception as e:
        print(f"✗ Failed to create VTK file: {e}")

if __name__ == "__main__":
    visualize_from_raw_data()
    create_paraview_compatible_vtk()
    
    print(f"\n=== SUMMARY ===")
    print("Files created:")
    import os
    files = ['poisson_visualization.png', 'poisson_simple.vtk']
    for fname in files:
        if os.path.exists(fname):
            size = os.path.getsize(fname)
            print(f"  ✓ {fname}: {size} bytes")
    
    print(f"\nTo try with ParaView:")
    print(f'  "/mnt/c/Program Files/ParaView 5.13.1/bin/paraview.exe" poisson_simple.vtk')
