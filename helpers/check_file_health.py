#!/usr/bin/env python3
# check_file_health.py
# Comprehensive health checker for FEniCSx output files

import os
import sys
import h5py
import xml.etree.ElementTree as ET
import json
import struct
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class FileHealthChecker:
    def __init__(self, output_dir="../output"):
        self.output_dir = Path(output_dir)
        self.issues = []
        self.summary = {}

    def log_issue(self, severity, file_path, message):
        """Log an issue with a file"""
        self.issues.append(
            {"severity": severity, "file": str(file_path), "message": message}
        )

    def check_file_exists_and_readable(self, file_path):
        """Basic file existence and readability check"""
        if not file_path.exists():
            self.log_issue("ERROR", file_path, "File does not exist")
            return False

        if not file_path.is_file():
            self.log_issue("ERROR", file_path, "Path is not a file")
            return False

        try:
            with open(file_path, "rb") as f:
                f.read(1)  # Try to read first byte
            return True
        except PermissionError:
            self.log_issue("ERROR", file_path, "Permission denied - cannot read file")
            return False
        except Exception as e:
            self.log_issue("ERROR", file_path, f"Cannot read file: {e}")
            return False

    def check_vtk_file(self, file_path):
        """Check VTK file health"""
        print(f"Checking VTK: {file_path.name}")

        if not self.check_file_exists_and_readable(file_path):
            return

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            if len(lines) < 4:
                self.log_issue("ERROR", file_path, "VTK file too short (< 4 lines)")
                return

            # Check VTK header
            if not lines[0].startswith("# vtk DataFile Version"):
                self.log_issue("WARNING", file_path, "Missing or invalid VTK header")

            # Check data format declaration
            format_line = None
            dataset_line = None
            points_line = None

            for i, line in enumerate(lines):
                line = line.strip()
                if line in ["ASCII", "BINARY"]:
                    format_line = i
                elif line.startswith("DATASET"):
                    dataset_line = i
                elif line.startswith("POINTS"):
                    points_line = i
                    # Extract number of points
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            num_points = int(parts[1])
                            self.summary[file_path.name] = {
                                "points": num_points,
                                "format": "VTK",
                            }
                        except ValueError:
                            self.log_issue(
                                "WARNING", file_path, "Invalid POINTS declaration"
                            )

            if format_line is None:
                self.log_issue("ERROR", file_path, "Missing ASCII/BINARY declaration")
            if dataset_line is None:
                self.log_issue("ERROR", file_path, "Missing DATASET declaration")
            if points_line is None:
                self.log_issue("ERROR", file_path, "Missing POINTS declaration")

            # Check for data sections
            has_point_data = any("POINT_DATA" in line for line in lines)
            has_cell_data = any("CELL_DATA" in line for line in lines)

            if not has_point_data and not has_cell_data:
                self.log_issue("WARNING", file_path, "No POINT_DATA or CELL_DATA found")

            print(f"  ✓ VTK file appears valid ({len(lines)} lines)")

        except UnicodeDecodeError:
            self.log_issue(
                "ERROR",
                file_path,
                "File contains non-UTF8 characters (possibly binary VTK)",
            )
        except Exception as e:
            self.log_issue("ERROR", file_path, f"Error reading VTK file: {e}")

    def check_xdmf_file(self, file_path):
        """Check XDMF file health"""
        print(f"Checking XDMF: {file_path.name}")

        if not self.check_file_exists_and_readable(file_path):
            return

        try:
            # Parse XML
            tree = ET.parse(file_path)
            root = tree.getroot()

            if root.tag != "Xdmf":
                self.log_issue(
                    "ERROR", file_path, f"Root element is {root.tag}, expected Xdmf"
                )
                return

            # Check version
            version = root.get("Version")
            if version:
                print(f"  XDMF Version: {version}")

            # Find domains, grids, and data items
            domains = root.findall(".//Domain")
            grids = root.findall(".//Grid")
            data_items = root.findall(".//DataItem")

            print(
                f"  Found: {len(domains)} domains, {len(grids)} grids, {len(data_items)} data items"
            )

            if len(domains) == 0:
                self.log_issue("ERROR", file_path, "No Domain elements found")
            if len(grids) == 0:
                self.log_issue("ERROR", file_path, "No Grid elements found")

            # Check data item references
            h5_files_referenced = set()
            for item in data_items:
                format_attr = item.get("Format", "")
                if format_attr == "HDF":
                    if item.text:
                        h5_ref = item.text.strip()
                        if ":" in h5_ref:
                            h5_file = h5_ref.split(":")[0]
                            h5_files_referenced.add(h5_file)

            # Check if referenced HDF5 files exist
            for h5_file in h5_files_referenced:
                h5_path = file_path.parent / h5_file
                if not h5_path.exists():
                    self.log_issue(
                        "ERROR", file_path, f"Referenced HDF5 file not found: {h5_file}"
                    )
                else:
                    print(f"  ✓ Found referenced HDF5: {h5_file}")

            self.summary[file_path.name] = {
                "domains": len(domains),
                "grids": len(grids),
                "data_items": len(data_items),
                "h5_refs": len(h5_files_referenced),
                "format": "XDMF",
            }

            print(f"  ✓ XDMF file appears valid")

        except ET.ParseError as e:
            self.log_issue("ERROR", file_path, f"XML parsing error: {e}")
        except Exception as e:
            self.log_issue("ERROR", file_path, f"Error reading XDMF file: {e}")

    def check_h5_file(self, file_path):
        """Check HDF5 file health"""
        print(f"Checking HDF5: {file_path.name}")

        if not self.check_file_exists_and_readable(file_path):
            return

        try:
            with h5py.File(file_path, "r") as f:
                # Get file info
                def count_items(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        return 1
                    return 0

                num_datasets = 0
                datasets_info = []

                def collect_info(name, obj):
                    nonlocal num_datasets
                    if isinstance(obj, h5py.Dataset):
                        num_datasets += 1
                        datasets_info.append(
                            {
                                "name": name,
                                "shape": obj.shape,
                                "dtype": str(obj.dtype),
                                "size": obj.size,
                            }
                        )

                f.visititems(collect_info)

                print(f"  Found {num_datasets} datasets")
                for info in datasets_info[:5]:  # Show first 5
                    print(
                        f"    {info['name']}: shape={info['shape']}, dtype={info['dtype']}"
                    )

                if len(datasets_info) > 5:
                    print(f"    ... and {len(datasets_info) - 5} more datasets")

                self.summary[file_path.name] = {
                    "datasets": num_datasets,
                    "format": "HDF5",
                }

                print(f"  ✓ HDF5 file appears valid")

        except Exception as e:
            self.log_issue("ERROR", file_path, f"Error reading HDF5 file: {e}")

    def check_pvd_file(self, file_path):
        """Check ParaView Data (PVD) file health"""
        print(f"Checking PVD: {file_path.name}")

        if not self.check_file_exists_and_readable(file_path):
            return

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            if root.tag != "VTKFile":
                self.log_issue(
                    "ERROR", file_path, f"Root element is {root.tag}, expected VTKFile"
                )
                return

            # Check for Collection and DataSet elements
            collections = root.findall(".//Collection")
            datasets = root.findall(".//DataSet")

            print(f"  Found: {len(collections)} collections, {len(datasets)} datasets")

            # Check referenced files
            missing_files = []
            for dataset in datasets:
                file_attr = dataset.get("file", "")
                if file_attr:
                    ref_path = file_path.parent / file_attr
                    if not ref_path.exists():
                        missing_files.append(file_attr)

            if missing_files:
                self.log_issue(
                    "ERROR", file_path, f"Referenced files not found: {missing_files}"
                )
            else:
                print(f"  ✓ All referenced files found")

            self.summary[file_path.name] = {
                "collections": len(collections),
                "datasets": len(datasets),
                "missing_refs": len(missing_files),
                "format": "PVD",
            }

        except ET.ParseError as e:
            self.log_issue("ERROR", file_path, f"XML parsing error: {e}")
        except Exception as e:
            self.log_issue("ERROR", file_path, f"Error reading PVD file: {e}")

    def check_bp_directory(self, dir_path):
        """Check ADIOS2 BP directory health"""
        print(f"Checking BP directory: {dir_path.name}")

        if not dir_path.exists():
            self.log_issue("ERROR", dir_path, "BP directory does not exist")
            return

        if not dir_path.is_dir():
            self.log_issue("ERROR", dir_path, "BP path is not a directory")
            return

        # Check for required BP files
        required_files = ["md.0", "md.idx"]
        optional_files = ["data.0", "mmd.0", "profiling.json"]

        found_files = [f.name for f in dir_path.iterdir() if f.is_file()]

        missing_required = [f for f in required_files if f not in found_files]
        found_optional = [f for f in optional_files if f in found_files]

        if missing_required:
            self.log_issue(
                "ERROR", dir_path, f"Missing required BP files: {missing_required}"
            )

        print(f"  Found files: {found_files}")
        print(
            f"  Required files present: {len(required_files) - len(missing_required)}/{len(required_files)}"
        )
        print(f"  Optional files present: {len(found_optional)}/{len(optional_files)}")

        # Try to read metadata if available
        md_file = dir_path / "md.0"
        if md_file.exists():
            try:
                size = md_file.stat().st_size
                print(f"  Metadata file size: {size} bytes")
                if size == 0:
                    self.log_issue("WARNING", dir_path, "Metadata file is empty")
            except Exception as e:
                self.log_issue("WARNING", dir_path, f"Cannot read metadata file: {e}")

        self.summary[dir_path.name] = {
            "files_found": len(found_files),
            "required_missing": len(missing_required),
            "format": "ADIOS2_BP",
        }

        if not missing_required:
            print(f"  ✓ BP directory appears valid")

    def run_comprehensive_check(self):
        """Run comprehensive health check on all output files"""
        print("=" * 80)
        print("FENICSX OUTPUT FILE HEALTH CHECK")
        print("=" * 80)
        print(f"Checking directory: {self.output_dir.absolute()}")
        print()

        if not self.output_dir.exists():
            print(f"ERROR: Output directory does not exist: {self.output_dir}")
            return

        # Get all files and directories
        all_items = list(self.output_dir.iterdir())

        # Categorize by type
        vtk_files = [f for f in all_items if f.suffix == ".vtk" and f.is_file()]
        xdmf_files = [f for f in all_items if f.suffix == ".xdmf" and f.is_file()]
        h5_files = [f for f in all_items if f.suffix == ".h5" and f.is_file()]
        pvd_files = [f for f in all_items if f.suffix == ".pvd" and f.is_file()]
        bp_dirs = [d for d in all_items if d.suffix == ".bp" and d.is_dir()]

        print(f"Found files:")
        print(f"  VTK files: {len(vtk_files)}")
        print(f"  XDMF files: {len(xdmf_files)}")
        print(f"  HDF5 files: {len(h5_files)}")
        print(f"  PVD files: {len(pvd_files)}")
        print(f"  BP directories: {len(bp_dirs)}")
        print()

        # Check each file type
        for vtk_file in vtk_files:
            self.check_vtk_file(vtk_file)
            print()

        for xdmf_file in xdmf_files:
            self.check_xdmf_file(xdmf_file)
            print()

        for h5_file in h5_files:
            self.check_h5_file(h5_file)
            print()

        for pvd_file in pvd_files:
            self.check_pvd_file(pvd_file)
            print()

        for bp_dir in bp_dirs:
            self.check_bp_directory(bp_dir)
            print()

        # Summary report
        self.print_summary_report()

    def print_summary_report(self):
        """Print summary report of all issues found"""
        print("=" * 80)
        print("HEALTH CHECK SUMMARY")
        print("=" * 80)

        # Count issues by severity
        errors = [issue for issue in self.issues if issue["severity"] == "ERROR"]
        warnings = [issue for issue in self.issues if issue["severity"] == "WARNING"]

        print(f"Total files checked: {len(self.summary)}")
        print(f"Errors found: {len(errors)}")
        print(f"Warnings found: {len(warnings)}")
        print()

        if errors:
            print("ERRORS:")
            for error in errors:
                print(f"  ❌ {error['file']}: {error['message']}")
            print()

        if warnings:
            print("WARNINGS:")
            for warning in warnings:
                print(f"  ⚠️  {warning['file']}: {warning['message']}")
            print()

        if not errors and not warnings:
            print("✅ No issues found! All files appear healthy.")

        # File summary table
        print("FILE SUMMARY:")
        print("-" * 60)
        for filename, info in self.summary.items():
            format_name = info.get("format", "Unknown")
            details = []

            if "points" in info:
                details.append(f"{info['points']} points")
            if "datasets" in info:
                details.append(f"{info['datasets']} datasets")
            if "grids" in info:
                details.append(f"{info['grids']} grids")
            if "files_found" in info:
                details.append(f"{info['files_found']} files")

            detail_str = ", ".join(details) if details else "No details"
            print(f"  {filename:<30} [{format_name:<8}] {detail_str}")

        print()
        print("RECOMMENDATIONS:")

        if any("thermal_stress" in filename for filename in self.summary.keys()):
            print("  • For thermal stress files, try opening individual VTK files:")
            print("    - thermal_field.vtk (temperature data)")
            print("    - displacement_field.vtk (displacement data)")

        if any(issue["file"].endswith(".xdmf") for issue in errors):
            print("  • XDMF files have issues - use VTK format instead")

        if any(issue["file"].endswith(".bp") for issue in errors):
            print("  • BP (ADIOS2) files may not be compatible with Windows ParaView")

        print("  • Use simple VTK format for best ParaView compatibility")
        print(
            "  • Check that all referenced files (like .h5) exist alongside .xdmf files"
        )

    def visualize_vtk_file(self, file_path):
        """Create a simple visualization of VTK file data"""
        print(f"Creating visualization for: {file_path.name}")

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Parse VTK file
            points = []
            scalars = []
            vectors = []
            reading_points = False
            reading_scalars = False
            reading_vectors = False

            for line in lines:
                line = line.strip()

                if line.startswith("POINTS"):
                    reading_points = True
                    parts = line.split()
                    num_points = int(parts[1])
                    continue
                elif line.startswith("SCALARS"):
                    reading_scalars = True
                    reading_points = False
                    continue
                elif line.startswith("VECTORS"):
                    reading_vectors = True
                    reading_points = False
                    reading_scalars = False
                    continue
                elif (
                    line.startswith("LOOKUP_TABLE")
                    or line.startswith("POINT_DATA")
                    or line.startswith("CELL_DATA")
                ):
                    continue
                elif any(
                    line.startswith(kw) for kw in ["DATASET", "CELLS", "CELL_TYPES"]
                ):
                    reading_points = False
                    reading_scalars = False
                    reading_vectors = False
                    continue

                # Read data
                if reading_points and line:
                    try:
                        coords = [float(x) for x in line.split()]
                        if len(coords) >= 2:
                            points.append([coords[0], coords[1]])
                    except ValueError:
                        pass
                elif reading_scalars and line:
                    try:
                        scalars.extend([float(x) for x in line.split()])
                    except ValueError:
                        pass
                elif reading_vectors and line:
                    try:
                        vector_data = [float(x) for x in line.split()]
                        if len(vector_data) >= 2:
                            vectors.append([vector_data[0], vector_data[1]])
                    except ValueError:
                        pass

            if not points:
                print("  No point data found to visualize")
                return

            points = np.array(points)

            # Create visualization
            fig, axes = plt.subplots(
                1, 2 if vectors else 1, figsize=(12 if vectors else 6, 5)
            )
            if not vectors:
                axes = [axes]

            # Plot 1: Points with scalars (if available)
            ax1 = axes[0]
            if scalars and len(scalars) == len(points):
                scatter = ax1.scatter(
                    points[:, 0], points[:, 1], c=scalars, cmap="viridis", s=20
                )
                plt.colorbar(scatter, ax=ax1)
                ax1.set_title(
                    f"Scalar Field\nRange: [{min(scalars):.3f}, {max(scalars):.3f}]"
                )
            else:
                ax1.scatter(points[:, 0], points[:, 1], s=20, alpha=0.6)
                ax1.set_title("Point Distribution")

            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_aspect("equal")
            ax1.grid(True, alpha=0.3)

            # Plot 2: Vector field (if available)
            if vectors and len(vectors) == len(points):
                ax2 = axes[1]
                vectors = np.array(vectors)

                # Subsample for clarity if too many points
                if len(points) > 100:
                    step = len(points) // 50
                    indices = slice(0, None, step)
                    plot_points = points[indices]
                    plot_vectors = vectors[indices]
                else:
                    plot_points = points
                    plot_vectors = vectors

                ax2.quiver(
                    plot_points[:, 0],
                    plot_points[:, 1],
                    plot_vectors[:, 0],
                    plot_vectors[:, 1],
                    scale_units="xy",
                    angles="xy",
                    scale=1,
                    alpha=0.7,
                )
                ax2.set_title("Vector Field")
                ax2.set_xlabel("X")
                ax2.set_ylabel("Y")
                ax2.set_aspect("equal")
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            output_name = f"{file_path.stem}_visualization.png"
            plt.savefig(output_name, dpi=150, bbox_inches="tight")
            print(f"  Visualization saved as: {output_name}")
            plt.show()

        except Exception as e:
            print(f"  Error creating visualization: {e}")

    def visualize_bp_directory(self, dir_path):
        """Attempt to visualize BP (ADIOS2) directory contents"""
        print(f"Creating visualization for BP directory: {dir_path.name}")

        try:
            print("  Attempting to import adios2...")
            import adios2

            print(
                f"  Available adios2 attributes: {[attr for attr in dir(adios2) if not attr.startswith('_')]}"
            )
            print(
                f"  ✓ adios2 imported successfully, version: {adios2.__version__ if hasattr(adios2, '__version__') else 'unknown'}"
            )

            print(f"  Attempting to open BP file: {str(dir_path)}")

            # Try using FileReader
            reader = adios2.FileReader(str(dir_path))

            # Get available variables
            variables = reader.available_variables()
            print(f"  Found variables: {list(variables.keys())}")

            if not variables:
                print("  No variables found in BP file")
                reader.close()
                return

            # Show variable details
            for var_name, var_info in variables.items():
                print(f"    {var_name}: {var_info}")

            # Read and visualize key variables
            key_vars = ["Temperature", "Displacement", "geometry"]
            available_key_vars = [v for v in key_vars if v in variables]

            if not available_key_vars:
                print("  No key variables (Temperature, Displacement, geometry) found")
                reader.close()
                return

            # Create plots
            fig, axes = plt.subplots(
                1, len(available_key_vars), figsize=(5 * len(available_key_vars), 5)
            )
            if len(available_key_vars) == 1:
                axes = [axes]

            for i, var_name in enumerate(available_key_vars):
                print(f"  Reading variable: {var_name}")
                try:
                    data = reader.read(var_name)
                    print(f"    Data shape: {data.shape}, dtype: {data.dtype}")
                    print(f"    Data range: [{data.min():.3f}, {data.max():.3f}]")

                    ax = axes[i]

                    if var_name == "geometry" and data.ndim == 2 and data.shape[1] >= 2:
                        # Plot mesh points
                        ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.6)
                        ax.set_title("Mesh Geometry")
                        ax.set_xlabel("X [m]")
                        ax.set_ylabel("Y [m]")
                        ax.set_aspect("equal")
                    elif data.ndim == 1:
                        ax.plot(data)
                        ax.set_title(
                            f"{var_name}\nRange: [{data.min():.3f}, {data.max():.3f}]"
                        )
                        ax.set_ylabel("Value")
                        ax.set_xlabel("Node Index")
                    else:
                        # For displacement (vector) or other multi-dimensional data
                        if data.ndim == 2:
                            im = ax.imshow(data, cmap="viridis", aspect="auto")
                            plt.colorbar(im, ax=ax)
                            ax.set_title(f"{var_name}\nShape: {data.shape}")
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                f"{var_name}\nShape: {data.shape}\nRange: [{data.min():.3f}, {data.max():.3f}]",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="round", facecolor="lightblue"),
                            )
                            ax.set_title(f"{var_name}")
                            ax.axis("off")

                except Exception as e:
                    print(f"    Error reading {var_name}: {e}")
                    ax = axes[i]
                    ax.text(
                        0.5,
                        0.5,
                        f"Error reading\n{var_name}\n{str(e)[:50]}...",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{var_name} (Error)")

            reader.close()

            plt.tight_layout()
            output_name = f"{dir_path.stem}_bp_simulation_data.png"
            plt.savefig(output_name, dpi=150, bbox_inches="tight")
            print(f"  ✓ Simulation data visualization saved as: {output_name}")
            plt.show()

        except ImportError as e:
            print(f"  Import error: {e}")
            print("  adios2 Python module not available - cannot read BP data directly")
            self._visualize_bp_metadata(dir_path)
        except Exception as e:
            print(f"  Error opening/reading BP file: {e}")
            print(f"  Error type: {type(e)}")
            import traceback

            traceback.print_exc()
            self._visualize_bp_metadata(dir_path)

    def _visualize_bp_metadata(self, dir_path):
        """Fallback visualization showing BP file structure"""
        print("  Creating metadata visualization...")

        try:
            files_info = []
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    size = file_path.stat().st_size
                    files_info.append((file_path.name, size))

            if not files_info:
                print("  No files found in BP directory")
                return

            # Create a simple bar chart of file sizes
            names, sizes = zip(*files_info)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # File sizes bar chart
            ax1.bar(names, sizes)
            ax1.set_title("BP Directory File Sizes")
            ax1.set_ylabel("Size (bytes)")
            ax1.tick_params(axis="x", rotation=45)

            # File structure info
            info_text = f"BP Directory: {dir_path.name}\n\n"
            info_text += f"Total files: {len(files_info)}\n"
            info_text += f"Total size: {sum(sizes)} bytes\n\n"
            info_text += "Files:\n"
            for name, size in files_info:
                info_text += f"  {name}: {size} bytes\n"

            ax2.text(
                0.05,
                0.95,
                info_text,
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
            )
            ax2.set_title("Directory Information")
            ax2.axis("off")

            plt.tight_layout()
            output_name = f"{dir_path.stem}_bp_metadata.png"
            plt.savefig(output_name, dpi=150, bbox_inches="tight")
            print(f"  Metadata visualization saved as: {output_name}")
            plt.show()

        except Exception as e:
            print(f"  Error creating metadata visualization: {e}")

    def check_vtu_file(self, file_path):
        """Check VTU (VTK Unstructured Grid) file health"""
        print(f"Checking VTU: {file_path.name}")

        if not self.check_file_exists_and_readable(file_path):
            return

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            if root.tag != "VTKFile":
                self.log_issue(
                    "ERROR", file_path, f"Root element is {root.tag}, expected VTKFile"
                )
                return

            # Check VTK file type
            file_type = root.get("type", "")
            if file_type != "UnstructuredGrid":
                self.log_issue(
                    "WARNING",
                    file_path,
                    f"File type is {file_type}, expected UnstructuredGrid",
                )

            # Find data arrays
            pieces = root.findall(".//Piece")
            points = root.findall(".//Points")
            cells = root.findall(".//Cells")
            point_data = root.findall(".//PointData")

            print(
                f"  Found: {len(pieces)} pieces, {len(points)} point sets, {len(cells)} cell sets"
            )
            print(f"  Data arrays: {len(point_data)} point data sections")

            self.summary[file_path.name] = {
                "pieces": len(pieces),
                "points": len(points),
                "cells": len(cells),
                "point_data": len(point_data),
                "format": "VTU",
            }

            print(f"  ✓ VTU file appears valid")

        except ET.ParseError as e:
            self.log_issue("ERROR", file_path, f"XML parsing error: {e}")
        except Exception as e:
            self.log_issue("ERROR", file_path, f"Error reading VTU file: {e}")

    def visualize_vtu_file(self, file_path):
        """Visualize VTU file data"""
        print(f"Creating visualization for VTU file: {file_path.name}")

        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(file_path)
            root = tree.getroot()

            # Extract point coordinates
            points_elem = root.find(".//Points/DataArray")
            if points_elem is not None and points_elem.text:
                coords_text = points_elem.text.strip()
                coords_flat = [float(x) for x in coords_text.split()]
                coords = np.array(coords_flat).reshape(-1, 3)[:, :2]  # Take only x,y

                # Extract point data arrays
                point_data_arrays = {}
                for data_array in root.findall(".//PointData/DataArray"):
                    name = data_array.get("Name", "Unknown")
                    if data_array.text:
                        values = [float(x) for x in data_array.text.strip().split()]
                        point_data_arrays[name] = np.array(values)

                # Create visualization
                fig, axes = plt.subplots(
                    1,
                    len(point_data_arrays) + 1,
                    figsize=(5 * (len(point_data_arrays) + 1), 5),
                )
                if len(point_data_arrays) == 0:
                    axes = [axes]

                # Plot mesh
                ax_mesh = axes[0]
                ax_mesh.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.6)
                ax_mesh.set_title("Mesh Points")
                ax_mesh.set_xlabel("X")
                ax_mesh.set_ylabel("Y")
                ax_mesh.set_aspect("equal")

                # Plot data arrays
                for i, (name, values) in enumerate(point_data_arrays.items()):
                    if i + 1 < len(axes):
                        ax = axes[i + 1]
                        scatter = ax.scatter(
                            coords[:, 0], coords[:, 1], c=values, cmap="viridis", s=20
                        )
                        plt.colorbar(scatter, ax=ax)
                        ax.set_title(f"{name}")
                        ax.set_xlabel("X")
                        ax.set_ylabel("Y")
                        ax.set_aspect("equal")

                plt.tight_layout()
                output_name = f"{file_path.stem}_vtu_visualization.png"
                plt.savefig(output_name, dpi=150, bbox_inches="tight")
                print(f"  Visualization saved as: {output_name}")
                plt.show()

        except Exception as e:
            print(f"  Error creating VTU visualization: {e}")

    def visualize_pvd_file(self, file_path):
        """Visualize PVD file by opening referenced VTU files"""
        print(f"Creating visualization for PVD file: {file_path.name}")

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Find referenced VTU files
            datasets = root.findall(".//DataSet")
            if not datasets:
                print("  No datasets found in PVD file")
                return

            # Visualize the first referenced file
            first_dataset = datasets[0]
            vtu_file = first_dataset.get("file", "")
            if vtu_file:
                vtu_path = file_path.parent / vtu_file
                if vtu_path.exists():
                    print(f"  Visualizing referenced file: {vtu_file}")
                    self.visualize_vtu_file(vtu_path)
                else:
                    print(f"  Referenced file not found: {vtu_file}")

        except Exception as e:
            print(f"  Error creating PVD visualization: {e}")

    def visualize_xdmf_file(self, file_path):
        """Visualize XDMF file by reading HDF5 data"""
        print(f"Creating visualization for XDMF file: {file_path.name}")

        try:
            # Parse XDMF to find HDF5 references
            tree = ET.parse(file_path)
            root = tree.getroot()

            h5_refs = {}
            for item in root.findall(".//DataItem"):
                if item.get("Format") == "HDF" and item.text:
                    h5_ref = item.text.strip()
                    if ":" in h5_ref:
                        h5_file, h5_path = h5_ref.split(":", 1)
                        if h5_file not in h5_refs:
                            h5_refs[h5_file] = []
                        h5_refs[h5_file].append(h5_path)

            if not h5_refs:
                print("  No HDF5 references found in XDMF")
                return

            # Read data from HDF5 files
            for h5_file, paths in h5_refs.items():
                h5_path = file_path.parent / h5_file
                if h5_path.exists():
                    self.visualize_h5_file(h5_path)
                    break

        except Exception as e:
            print(f"  Error creating XDMF visualization: {e}")

    def visualize_h5_file(self, file_path):
        """Visualize HDF5 file data"""
        print(f"Creating visualization for HDF5 file: {file_path.name}")

        try:
            with h5py.File(file_path, "r") as f:
                datasets = {}

                def collect_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        datasets[name] = obj[...]

                f.visititems(collect_datasets)

                if not datasets:
                    print("  No datasets found in HDF5 file")
                    return

                # Look for key datasets
                geometry_data = None
                function_data = {}

                for name, data in datasets.items():
                    if "geometry" in name.lower():
                        geometry_data = data
                    elif "function" in name.lower() or name.endswith("/0"):
                        func_name = name.split("/")[-2] if "/" in name else name
                        function_data[func_name] = data

                # Create visualization
                num_plots = 1 + len(function_data)
                fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
                if num_plots == 1:
                    axes = [axes]

                # Plot geometry if available
                if geometry_data is not None and geometry_data.ndim == 2:
                    ax = axes[0]
                    ax.scatter(
                        geometry_data[:, 0], geometry_data[:, 1], s=10, alpha=0.6
                    )
                    ax.set_title("Geometry")
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_aspect("equal")

                # Plot function data
                plot_idx = 1
                for func_name, data in function_data.items():
                    if plot_idx < len(axes) and geometry_data is not None:
                        ax = axes[plot_idx]
                        if data.ndim == 2 and data.shape[1] == 1:
                            data = data.flatten()

                        if data.ndim == 1 and len(data) == len(geometry_data):
                            scatter = ax.scatter(
                                geometry_data[:, 0],
                                geometry_data[:, 1],
                                c=data,
                                cmap="viridis",
                                s=20,
                            )
                            plt.colorbar(scatter, ax=ax)
                            ax.set_title(f"{func_name}")
                            ax.set_xlabel("X")
                            ax.set_ylabel("Y")
                            ax.set_aspect("equal")
                        plot_idx += 1

                plt.tight_layout()
                output_name = f"{file_path.stem}_h5_visualization.png"
                plt.savefig(output_name, dpi=150, bbox_inches="tight")
                print(f"  Visualization saved as: {output_name}")
                plt.show()

        except Exception as e:
            print(f"  Error creating HDF5 visualization: {e}")


def main():
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])

        if target_path.is_file():
            print(f"DEBUG: Detected as file, suffix: {target_path.suffix}")
        elif target_path.is_dir():
            print(f"DEBUG: Detected as directory, suffix: {target_path.suffix}")
        else:
            print(f"DEBUG: Path doesn't exist or unrecognized type")

        checker = FileHealthChecker(".")  # Dummy directory

        if target_path.is_file():
            # Check individual file
            print("=" * 80)
            print("FENICSX FILE HEALTH CHECK")
            print("=" * 80)
            print(f"Checking file: {target_path.absolute()}")
            print()

            # Determine file type and check accordingly
            if target_path.suffix == ".vtk":
                checker.check_vtk_file(target_path)
            elif target_path.suffix == ".xdmf":
                checker.check_xdmf_file(target_path)
            elif target_path.suffix == ".h5":
                checker.check_h5_file(target_path)
            elif target_path.suffix == ".pvd":
                checker.check_pvd_file(target_path)
            elif target_path.suffix == ".vtu":
                checker.check_vtu_file(target_path)
            else:
                print(f"Unsupported file type: {target_path.suffix}")
                return

        elif target_path.is_dir():
            # Handle directories - could be BP or regular directory scan
            if target_path.suffix == ".bp":
                # Check individual BP directory
                print("=" * 80)
                print("FENICSX FILE HEALTH CHECK")
                print("=" * 80)
                print(f"Checking BP directory: {target_path.absolute()}")
                print()

                checker.check_bp_directory(target_path)
            else:
                # Check entire directory
                checker = FileHealthChecker(target_path)
                checker.run_comprehensive_check()
                return
        else:
            print(f"ERROR: Path does not exist: {target_path}")
            return

        # Print results for single item (moved outside the conditionals)
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)

        if checker.issues:
            for issue in checker.issues:
                severity_icon = "❌" if issue["severity"] == "ERROR" else "⚠️"
                print(f"{severity_icon} {issue['severity']}: {issue['message']}")
        else:
            print("✅ File/directory appears healthy - no issues found!")

            # Offer visualization based on file type
            if target_path.is_file() and target_path.suffix == ".vtk":
                print("\n📊 Creating VTK visualization...")
                checker.visualize_vtk_file(target_path)
            elif target_path.is_file() and target_path.suffix == ".vtu":
                print("\n📊 Creating VTU visualization...")
                checker.visualize_vtu_file(target_path)
            elif target_path.is_file() and target_path.suffix == ".pvd":
                print("\n📊 Creating PVD visualization...")
                checker.visualize_pvd_file(target_path)
            elif target_path.is_file() and target_path.suffix == ".xdmf":
                print("\n📊 Creating XDMF visualization...")
                checker.visualize_xdmf_file(target_path)
            elif target_path.is_file() and target_path.suffix == ".h5":
                print("\n📊 Creating HDF5 visualization...")
                checker.visualize_h5_file(target_path)
            elif target_path.is_dir() and target_path.suffix == ".bp":
                print("\n📊 Creating BP visualization...")
                checker.visualize_bp_directory(target_path)
    else:
        # Default behavior - check output directory
        checker = FileHealthChecker("../output")
        checker.run_comprehensive_check()


if __name__ == "__main__":
    main()
