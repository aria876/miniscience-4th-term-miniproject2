#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pv_quicklook_gui.py
A monolithic, cross-platform quicklook GUI for visualizing simulation outputs via the ParaView Python API.

Usage:
    python pv_quicklook_gui.py <path-to-data-file>

Features:
- Accepts a path to any supported dataset (.xdmf, .pvd, .pvtu, .vtu, .vtk, some .h5 with companion .xdmf, ADIOS2 .bp if available).
- Simple, focused GUI (Tkinter) that avoids the complexity of the ParaView GUI.
- Lets the user set the path to the ParaView desktop executable (used only if you want to "Open in ParaView" externally).
- Loads the dataset with paraview.simple, lists available arrays, lets you pick coloring and timestep.
- Renders offscreen via ParaView and shows a live preview inside the GUI (PNG snapshots).
- Basic camera controls (reset, azimuth, elevation), representation modes, color rescale, and screenshot export.

Notes:
- You should run this with a Python that can import paraview.simple (e.g., pvpython), or a Python where ParaView’s Python modules are on PYTHONPATH.
- If paraview.simple is not importable, the app will still launch and guide you to run it with pvpython or fix PYTHONPATH.
- The ParaView executable path you set here is optional and only used for the "Open in ParaView GUI" button.

Tested with ParaView 5.11–5.13. Minor adjustments may be needed for other versions.

Author: (you)
License: MIT
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import json
import shutil
import platform
import subprocess
import argparse
import traceback
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import argparse
import traceback
import textwrap

# -------------------------
# Safe import of paraview.simple
# -------------------------
PARAVIEW_AVAILABLE = True
try:
    # Silence ParaView's verbose startup output a bit
    os.environ.setdefault("PV_PLUGIN_PATH", "")
    from paraview.simple import (
        GetAnimationScene,
        GetTimeKeeper,
        XDMFReader,
        PVDReader,
        LegacyVTKReader,
        XMLUnstructuredGridReader,
        XMLPartitionedUnstructuredGridReader,
        CreateView,
        Render,
        Show,
        Hide,
        ResetCamera,
        GetActiveViewOrCreate,
        GetColorTransferFunction,
        GetOpacityTransferFunction,
        ColorBy,
        GetDisplayProperties,
        Slice,
        Contour,
        ExtractSurface,
        Calculator,
        SaveScreenshot,
        HideAll,
        GetSources,
        SetActiveSource,
        GetActiveSource,
        CreateRenderView,
        GetRenderView,
    )

    # Some readers/filters may not exist depending on build/plugins
    try:
        from paraview.simple import ADIOS2VTXReader  # Optional
    except Exception:
        ADIOS2VTXReader = None
except Exception as e:
    PARAVIEW_AVAILABLE = False
    PARAVIEW_IMPORT_ERROR = str(e)
    # Capture a full traceback string for verbose diagnostics
    try:
        PARAVIEW_IMPORT_TRACEBACK = traceback.format_exc()
    except Exception:
        PARAVIEW_IMPORT_TRACEBACK = PARAVIEW_IMPORT_ERROR

# -------------------------
# Tkinter GUI
# -------------------------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_NAME = "PV QuickLook"
DEFAULT_PARAVIEW_WIN = r"C:\Program Files\ParaView 5.13.1\bin\paraview.exe"

# Supported file patterns (not exhaustive)
SUPPORTED_EXTS = {
    ".xdmf": "XDMFReader",
    ".xmf": "XDMFReader",
    ".pvd": "PVDReader",
    ".pvtu": "XMLPartitionedUnstructuredGridReader",
    ".vtu": "XMLUnstructuredGridReader",
    ".vtk": "LegacyVTKReader",
    ".bp": "ADIOS2VTXReader",  # if available
    ".h5": "HDF5 (via companion XDMF if found)",
    ".hdf5": "HDF5 (via companion XDMF if found)",
    ".npz": "NumPy (metadata view only)",
}


def human_path(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def which_exe(cmd: str) -> Optional[str]:
    """Cross-platform shutil.which wrapper returning absolute path or None."""
    out = shutil.which(cmd)
    return out


def infer_pvpython_from_gui_path(gui_path: str) -> Optional[str]:
    r"""Given ...\bin\paraview.exe, try to return ...\bin\pvpython.exe (quote-safe)."""
    try:
        s = (gui_path or "").strip().strip('"').strip("'")
        if not s:
            return None
        p = Path(s)
        # Windows: paraview.exe → pvpython.exe
        if p.name.lower().startswith("paraview") and p.suffix.lower() == ".exe":
            cand = p.with_name("pvpython.exe")
            if cand.exists():
                return str(cand)
        # POSIX: paraview → pvpython
        if p.name.lower() == "paraview":
            cand = p.with_name("pvpython")
            if cand.exists():
                return str(cand)
    except Exception:
        pass
    return None


# -------------------------
# ParaView helpers
# -------------------------


class PVSession:
    """
    Minimal manager around ParaView 'simple' to load a dataset and render PNG snapshots.
    """

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.view = None
        self.source = None
        self.display = None
        self.anim = None
        self.timekeeper = None
        self.tmpdir = Path(tempfile.mkdtemp(prefix="pvquicklook_"))
        self.last_png = None
        self._time_steps = []
        self._array_info = []  # list of tuples (association, name)
        self._repr_modes = ["Surface", "Surface With Edges", "Wireframe", "Points"]
        self._current_array = None  # ("POINTS"|"CELLS", "name")
        self._dataset_basename = self.data_path.name

    def cleanup(self):
        try:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        except Exception:
            pass

    # -------- Loading & readers --------
    def _guess_reader(self):
        suffix = self.data_path.suffix.lower()
        if suffix in (".h5", ".hdf5"):
            # Attempt to locate a sibling .xdmf
            xd = self._find_companion_xdmf()
            if xd is not None:
                return "XDMFReader", xd
            return None, None
        if suffix in (".xdmf", ".xmf"):
            return "XDMFReader", self.data_path
        if suffix == ".pvd":
            return "PVDReader", self.data_path
        if suffix == ".pvtu":
            return "XMLPartitionedUnstructuredGridReader", self.data_path
        if suffix == ".vtu":
            return "XMLUnstructuredGridReader", self.data_path
        if suffix == ".vtk":
            return "LegacyVTKReader", self.data_path
        if suffix == ".bp":
            if ADIOS2VTXReader is not None:
                return "ADIOS2VTXReader", self.data_path
            return None, None
        if suffix == ".npz":
            # We'll not load with ParaView; just metadata and image disabled
            return "NPZ", self.data_path
        return None, None

    def _find_companion_xdmf(self) -> Optional[Path]:
        stem = self.data_path.stem
        parent = self.data_path.parent
        for alt in [stem + ".xdmf", stem + ".xmf"]:
            cand = parent / alt
            if cand.exists():
                return cand
        # Also try to find any xdmf in same folder that references this
        # (best effort: just return first .xdmf)
        for cand in parent.glob("*.xdmf"):
            return cand
        for cand in parent.glob("*.xmf"):
            return cand
        return None

    def _make_reader(self, reader_name: str, path: Path):
        if reader_name == "XDMFReader":
            r = XDMFReader(FileNames=[human_path(path)])
            r.ReadAllGeometry = 1
            r.ReadAllArrays = 1
            return r
        if reader_name == "PVDReader":
            return PVDReader(FileName=human_path(path))
        if reader_name == "LegacyVTKReader":
            return LegacyVTKReader(FileNames=[human_path(path)])
        if reader_name == "XMLUnstructuredGridReader":
            return XMLUnstructuredGridReader(FileName=human_path(path))
        if reader_name == "XMLPartitionedUnstructuredGridReader":
            return XMLPartitionedUnstructuredGridReader(FileName=human_path(path))
        if reader_name == "ADIOS2VTXReader" and ADIOS2VTXReader is not None:
            return ADIOS2VTXReader(FileName=human_path(path))
        return None

    def load(self) -> Tuple[bool, str]:
        if not PARAVIEW_AVAILABLE:
            return False, f"ParaView Python API not available: {PARAVIEW_IMPORT_ERROR}"
        reader_name, real_path = self._guess_reader()
        if reader_name is None:
            return (
                False,
                f"Unsupported or ambiguous file type for '{self.data_path.name}'. Try opening a .xdmf/.xmf, .pvd, .pvtu/.vtu, .vtk, or .bp (if ADIOS2VTXReader is available). For .h5, include a companion .xdmf.",
            )

        if reader_name == "NPZ":
            # We won't render NPZ; just metadata
            self.source = None
            return True, "Loaded .npz metadata mode (no ParaView rendering)."

        self.source = self._make_reader(reader_name, real_path)
        if self.source is None:
            return False, f"Failed to create reader '{reader_name}'."

        # Create / get view
        self.view = GetActiveViewOrCreate("RenderView")
        # initial show
        self.display = Show(self.source, self.view)

        # Initialize animation/time
        self.anim = GetAnimationScene()
        self.anim.UpdateAnimationUsingDataTimeSteps()
        self.timekeeper = GetTimeKeeper()

        # Reset camera and first render
        ResetCamera(self.view)
        Render(self.view)

        # Collect time steps
        self._time_steps = (
            list(self.timekeeper.TimestepValues)
            if hasattr(self.timekeeper, "TimestepValues")
            else []
        )
        # Collect arrays
        self._array_info = self._collect_arrays()

        # Default representation mode
        self.set_representation("Surface")

        # Default coloring
        if self._array_info:
            first_assoc, first_name = self._array_info[0]
            self.set_coloring(first_assoc, first_name)
        else:
            # solid color
            ColorBy(self.display, None)

        return True, f"Loaded with reader: {reader_name}"

    def _collect_arrays(self) -> List[Tuple[str, str]]:
        """
        Returns list of (association, name), where association is one of 'POINTS' or 'CELLS'.
        """
        out = []
        try:
            # ParaView uses tuple (association, arrayName) with ColorBy
            # Let's probe default display properties for available arrays
            # Using GetDisplayProperties(self.source). Representations keep list in a proxy info.
            # A more reliable way is to inspect the 'PointArrayStatus'/'CellArrayStatus' properties if present.
            info = self.source.GetProperty("PointArrayStatus")
            if info is not None and hasattr(info, "GetData"):
                names = info.GetData()
                for i in range(0, len(names), 2):
                    # names[i] = array name, names[i+1] = status (0/1)
                    nm = names[i]
                    out.append(("POINTS", nm))
            info = self.source.GetProperty("CellArrayStatus")
            if info is not None and hasattr(info, "GetData"):
                names = info.GetData()
                for i in range(0, len(names), 2):
                    nm = names[i]
                    out.append(("CELLS", nm))
        except Exception:
            pass
        # Deduplicate preserving order
        seen = set()
        uniq = []
        for a in out:
            if a not in seen:
                uniq.append(a)
                seen.add(a)
        return uniq

    # -------- Rendering & snapshot --------
    def snapshot(
        self, width: int = 1024, height: int = 768, quality: int = 95
    ) -> Optional[Path]:
        """Render offscreen PNG and return its path."""
        if self.view is None:
            return None
        png_path = self.tmpdir / f"frame_{int(time.time()*1000)}.png"
        # Ensure current render
        Render(self.view)
        SaveScreenshot(
            human_path(png_path),
            self.view,
            ImageResolution=[width, height],
            CompressionLevel=0,
            Quality=quality,
        )
        self.last_png = png_path
        return png_path

    # -------- Controls --------
    def get_time_steps(self) -> List[float]:
        return self._time_steps

    def set_time_step_index(self, idx: int) -> None:
        if not self._time_steps:
            return
        idx = max(0, min(idx, len(self._time_steps) - 1))
        t = self._time_steps[idx]
        try:
            self.timekeeper.Time = t
        except Exception:
            # Fallback via animation scene
            try:
                self.anim.AnimationTime = t
            except Exception:
                pass
        Render(self.view)

    def set_representation(self, mode: str) -> None:
        if self.display is None:
            return
        mode = mode or "Surface"
        dp = GetDisplayProperties(self.source, view=self.view)
        dp.SetRepresentationType(mode)
        Render(self.view)

    def get_representation_modes(self) -> List[str]:
        return list(self._repr_modes)

    def arrays(self) -> List[Tuple[str, str]]:
        return list(self._array_info)

    def set_coloring(self, association: str, array_name: str) -> None:
        if self.display is None or self.source is None:
            return
        self._current_array = (association, array_name)
        try:
            assoc_key = "POINTS" if association.upper().startswith("POINT") else "CELLS"
            ColorBy(self.display, (assoc_key, array_name))
            # Rescale to data range
            self.rescale_color_to_data()
        except Exception:
            # fallback to solid color
            ColorBy(self.display, None)
        Render(self.view)

    def rescale_color_to_data(self):
        if self.display is None:
            return
        try:
            lut = (
                GetColorTransferFunction(self._current_array[1])
                if self._current_array
                else None
            )
            if lut is not None and hasattr(
                self.display, "RescaleTransferFunctionToDataRange"
            ):
                self.display.RescaleTransferFunctionToDataRange(True, False)
        except Exception:
            pass
        Render(self.view)

    def camera_reset(self):
        if self.view is None:
            return
        ResetCamera(self.view)
        Render(self.view)

    def camera_azimuth(self, deg: float = 15.0):
        if self.view is None:
            return
        try:
            cam = self.view.GetActiveCamera()
            cam.Azimuth(deg)
        except Exception:
            # Some PV builds: Use view.Azimuth
            try:
                self.view.Azimuth(deg)
            except Exception:
                pass
        Render(self.view)

    def camera_elevation(self, deg: float = 15.0):
        if self.view is None:
            return
        try:
            cam = self.view.GetActiveCamera()
            cam.Elevation(deg)
        except Exception:
            try:
                self.view.Elevation(deg)
            except Exception:
                pass
        Render(self.view)

    def export_screenshot(
        self, out_path: Path, w: int = 1920, h: int = 1080, quality: int = 95
    ) -> Tuple[bool, str]:
        if self.view is None:
            return False, "Nothing loaded."
        try:
            SaveScreenshot(
                human_path(out_path),
                self.view,
                ImageResolution=[w, h],
                CompressionLevel=0,
                Quality=quality,
            )
            return True, f"Saved screenshot to: {human_path(out_path)}"
        except Exception as e:
            return False, f"Failed to save screenshot: {e}"


class PVSubprocessSession:
    """
    Fallback session that renders via `pvpython` subprocesses.
    Slower than an in-process PVSession, but robust across Python versions.
    """

    def __init__(self, data_path: Path, pvpython_exe: Optional[str]):
        self.data_path = data_path
        self.pvpython = (
            pvpython_exe or which_exe("pvpython") or which_exe("pvpython.exe")
        )
        self.tmpdir = Path(tempfile.mkdtemp(prefix="pvquicklook_sub_"))
        self.view = True  # sentinel so GUI treats it as loaded
        self.source = True  # sentinel
        self.display = True  # sentinel
        self._time_steps: List[float] = []
        self._array_info: List[Tuple[str, str]] = []
        self._repr = "Surface"
        self._current_array: Optional[Tuple[str, str]] = None
        self._time_idx = -1
        self._last_err = ""

    def cleanup(self):
        try:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        except Exception:
            pass

    def _write_helper_script(self) -> Path:
        # Use the worker.py that you placed next to pv_quicklook_gui.py
        return Path(__file__).with_name("worker.py")

    def _pvpython(self) -> Optional[str]:
        return self.pvpython

    def load(self) -> Tuple[bool, str]:
        exe = self._pvpython()
        if not exe or not Path(exe).exists():
            return (
                False,
                "pvpython not found. Set the ParaView GUI path (we will infer pvpython.exe) or add pvpython to PATH.",
            )
        helper = self._write_helper_script()
        probe_json = self.tmpdir / "probe.json"
        cmd = [
            exe,
            str(helper),
            "--data",
            human_path(self.data_path),
            "--probe_json",
            str(probe_json),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if proc.returncode != 0:
                # surface stderr and a hint about pvpython path
                err = (proc.stderr or "").strip()
                hint = f"pvpython: {exe}"
                return (
                    False,
                    f"pvpython probe failed (exit {proc.returncode}). {hint}\n{err}",
                )
            meta = json.loads(probe_json.read_text(encoding="utf-8"))
            self._array_info = [(a[0], a[1]) for a in meta.get("arrays", [])]
            self._time_steps = meta.get("timesteps", [])
            if self._array_info:
                self._current_array = self._array_info[0]
            return (
                True,
                f"Loaded via pvpython subprocess (reader: {meta.get('reader','?')}).",
            )
        except FileNotFoundError:
            return False, f"pvpython not found at: {exe}"
        except Exception as e:
            return False, f"Subprocess error: {e}"

    def get_time_steps(self) -> List[float]:
        return list(self._time_steps)

    def last_error(self) -> str:
        return self._last_err

    def set_time_step_index(self, idx: int) -> None:
        if not self._time_steps:
            self._time_idx = -1
            return
        self._time_idx = max(0, min(idx, len(self._time_steps) - 1))

    def get_representation_modes(self) -> List[str]:
        return ["Surface", "Surface With Edges", "Wireframe", "Points"]

    def set_representation(self, mode: str) -> None:
        self._repr = mode or "Surface"

    def arrays(self) -> List[Tuple[str, str]]:
        return list(self._array_info)

    def set_coloring(self, association: str, array_name: str) -> None:
        self._current_array = (association, array_name)

    def rescale_color_to_data(self):
        # handled per-call in subprocess
        pass

    def camera_reset(self):
        pass

    def camera_azimuth(self, deg: float):
        pass

    def camera_elevation(self, deg: float):
        pass

    def snapshot(
        self, width: int = 1024, height: int = 768, quality: int = 95
    ) -> Optional[Path]:
        exe = self._pvpython()
        if not exe or not Path(exe).exists():
            return None
        helper = self._write_helper_script()
        out_png = self.tmpdir / f"frame_{int(time.time()*1000)}.png"
        args = [
            exe,
            str(helper),
            "--data",
            human_path(self.data_path),
            "--out_png",
            str(out_png),
            "--width",
            str(width),
            "--height",
            str(height),
            "--quality",
            str(quality),
            "--repr",
            self._repr or "Surface",
            "--rescale_colors",
            "--cam_reset",
        ]
        if self._current_array:
            assoc, name = self._current_array
            args += ["--color_assoc", assoc, "--color_array", name]
        if self._time_idx >= 0:
            args += ["--time_index", str(self._time_idx)]
        try:
            proc = subprocess.run(
                args, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if proc.returncode != 0:
                self._last_err = (
                    proc.stderr or ""
                ).strip() or f"pvpython exit {proc.returncode}"
                # keep a small breadcrumbs file with the last error
                (self.tmpdir / "last_snapshot_stderr.txt").write_text(
                    self._last_err, encoding="utf-8"
                )
                return None
            self._last_err = ""
            return out_png if out_png.exists() else None
        except Exception as e:
            self._last_err = str(e)
            (self.tmpdir / "last_snapshot_error.txt").write_text(
                self._last_err, encoding="utf-8"
            )
            return None

    def export_screenshot(
        self, out_path: Path, w: int = 1920, h: int = 1080, quality: int = 95
    ) -> Tuple[bool, str]:
        png = self.snapshot(w, h, quality)
        if not png or not png.exists():
            return False, "Failed to render screenshot via pvpython."
        try:
            shutil.copyfile(png, out_path)
            return True, f"Saved screenshot to: {human_path(out_path)}"
        except Exception as e:
            return False, f"Copy failed: {e}"


# -------------------------
# Tkinter Application
# -------------------------


class PVQuickLookApp(tk.Tk):
    def __init__(self, initial_path: Optional[Path] = None):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1100x800")
        self.minsize(1024, 700)

        self.paraview_exe_var = tk.StringVar()
        self.data_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Welcome!")

        # Default ParaView path (Windows)
        if platform.system().lower().startswith("win"):
            self.paraview_exe_var.set(DEFAULT_PARAVIEW_WIN)

        if initial_path:
            self.data_path_var.set(human_path(initial_path))

        # ParaView session
        self.session: Optional[PVSession] = None

        # Tk image cache for preview
        self.preview_img = None

        self._build_ui()

        # If initial path passed, auto-load
        if initial_path is not None and initial_path.exists():
            self.load_dataset()

    # ---- UI Layout ----
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Top bar: ParaView exe and dataset path
        top = ttk.Frame(self, padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        top.columnconfigure(4, weight=1)

        ttk.Label(top, text="ParaView GUI (optional):").grid(
            row=0, column=0, sticky="w"
        )
        pv_entry = ttk.Entry(top, textvariable=self.paraview_exe_var)
        pv_entry.grid(row=0, column=1, sticky="ew", padx=(6, 8))
        ttk.Button(top, text="Browse...", command=self._browse_paraview_exe).grid(
            row=0, column=2, padx=(0, 12)
        )
        ttk.Button(
            top, text="Open in ParaView GUI", command=self._open_in_paraview_gui
        ).grid(row=0, column=3)

        ttk.Label(top, text="Dataset:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        data_entry = ttk.Entry(top, textvariable=self.data_path_var)
        data_entry.grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=(6, 8), pady=(8, 0)
        )
        ttk.Button(top, text="Choose File...", command=self._browse_data).grid(
            row=1, column=4, sticky="e", pady=(8, 0)
        )
        ttk.Button(top, text="Load", command=self.load_dataset).grid(
            row=1, column=5, sticky="e", padx=(8, 0), pady=(8, 0)
        )

        # Status line
        status = ttk.Label(
            self, textvariable=self.status_var, anchor="w", padding=(10, 4)
        )
        status.grid(row=2, column=0, sticky="ew")

        # Main split: left controls, right preview
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        # Left panel (controls)
        left = ttk.Frame(main, padding=10)
        left.columnconfigure(1, weight=1)

        # Representation
        ttk.Label(left, text="Representation").grid(row=0, column=0, sticky="w")
        self.repr_var = tk.StringVar(value="Surface")
        self.repr_combo = ttk.Combobox(
            left, textvariable=self.repr_var, state="readonly"
        )
        self.repr_combo.grid(row=0, column=1, sticky="ew", pady=(0, 8))
        self.repr_combo.bind(
            "<<ComboboxSelected>>", lambda e: self._apply_representation()
        )

        # Coloring
        ttk.Label(left, text="Color by").grid(row=1, column=0, sticky="w")
        self.array_var = tk.StringVar()
        self.array_combo = ttk.Combobox(
            left, textvariable=self.array_var, state="readonly"
        )
        self.array_combo.grid(row=1, column=1, sticky="ew", pady=(0, 8))
        self.array_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_coloring())

        # Time step slider
        ttk.Label(left, text="Time step").grid(row=2, column=0, sticky="w")
        self.time_var = tk.IntVar(value=0)
        self.time_slider = ttk.Scale(
            left, from_=0, to=0, orient=tk.HORIZONTAL, command=self._on_time_slider
        )
        self.time_slider.grid(row=2, column=1, sticky="ew")

        self.time_idx_label = ttk.Label(left, text="0 / 0")
        self.time_idx_label.grid(row=3, column=1, sticky="e", pady=(2, 10))

        # Camera controls
        cam_frame = ttk.LabelFrame(left, text="Camera")
        cam_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 6))
        cam_frame.columnconfigure(0, weight=1)
        cam_frame.columnconfigure(1, weight=1)
        ttk.Button(cam_frame, text="Reset", command=self._camera_reset).grid(
            row=0, column=0, sticky="ew", padx=4, pady=4
        )
        ttk.Button(
            cam_frame, text="Azimuth +15°", command=lambda: self._camera_azimuth(+15)
        ).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(
            cam_frame,
            text="Elevation +15°",
            command=lambda: self._camera_elevation(+15),
        ).grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(
            cam_frame, text="Azimuth -15°", command=lambda: self._camera_azimuth(-15)
        ).grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        # Color scale
        ttk.Button(
            left, text="Rescale Colors to Data", command=self._rescale_colors
        ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 8))

        # Export
        exp_frame = ttk.LabelFrame(left, text="Export")
        exp_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 6))
        ttk.Button(
            exp_frame, text="Save Screenshot…", command=self._export_screenshot
        ).grid(row=0, column=0, sticky="ew", padx=4, pady=6)

        # Help / About
        help_frame = ttk.Frame(left)
        help_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(6, 6))
        ttk.Button(
            help_frame, text="Supported Formats…", command=self._show_supported
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(help_frame, text="About", command=self._about).grid(
            row=0, column=1, sticky="e"
        )

        # Right panel (preview)
        right = ttk.Frame(main, padding=(8, 8))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(right, bg="#222", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Add panels to paned window
        main.add(left, weight=0)
        main.add(right, weight=1)

        # Bind resize to refresh preview
        self.canvas.bind("<Configure>", lambda e: self._refresh_preview_async())

    # ---- Browsers & dialogs ----
    def _browse_paraview_exe(self):
        title = "Locate ParaView executable"
        if platform.system().lower().startswith("win"):
            filetypes = [("ParaView", "paraview.exe"), ("All files", "*.*")]
        elif platform.system().lower() == "darwin":
            filetypes = [("ParaView", "paraview"), ("All files", "*.*")]
        else:
            filetypes = [("ParaView", "paraview"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if path:
            self.paraview_exe_var.set(path)

    def _browse_data(self):
        title = "Open dataset"
        filetypes = [
            (
                "All supported",
                "*.xdmf *.xmf *.pvd *.pvtu *.vtu *.vtk *.bp *.h5 *.hdf5 *.npz",
            ),
            ("XDMF", "*.xdmf *.xmf"),
            ("PVD", "*.pvd"),
            ("VTK", "*.vtk"),
            ("VTU/PVTU", "*.vtu *.pvtu"),
            ("ADIOS2", "*.bp"),
            ("HDF5", "*.h5 *.hdf5"),
            ("NumPy", "*.npz"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if path:
            self.data_path_var.set(path)

    # ---- Load dataset ----
    def load_dataset(self):
        # Clean old session
        if self.session is not None:
            try:
                self.session.cleanup()
            except Exception:
                pass
            self.session = None

        data_path = Path(self.data_path_var.get().strip('"').strip())
        if not data_path.exists():
            self._set_status(f"File not found: {human_path(data_path)}", error=True)
            return

        if PARAVIEW_AVAILABLE:
            self.session = PVSession(data_path)
        else:
            # Subprocess fallback using pvpython
            pvpy = self._resolve_pvpython_exe()
            self.session = PVSubprocessSession(data_path, pvpy)

        ok, msg = self.session.load()
        self._set_status(msg, error=not ok)
        if not ok:
            if not PARAVIEW_AVAILABLE:
                messagebox.showerror(
                    "pvpython not found",
                    "Could not locate 'pvpython'.\n\n"
                    "Set the ParaView GUI path at the top (we will infer pvpython.exe), "
                    "or add the ParaView 'bin' folder to your PATH.",
                )
            return
        self._set_status(msg, error=not ok)
        if not ok:
            return

        # Populate representation modes
        try:
            self.repr_combo["values"] = self.session.get_representation_modes()
            self.repr_combo.set("Surface")
        except Exception:
            pass

        # Populate array list
        arrs = self.session.arrays()
        if arrs:
            arr_choices = [f"{assoc}:{name}" for assoc, name in arrs]
            self.array_combo["values"] = arr_choices
            self.array_combo.set(arr_choices[0])
        else:
            self.array_combo["values"] = []
            self.array_combo.set("")

        # Setup time slider
        tsteps = self.session.get_time_steps()
        if tsteps:
            self.time_slider.configure(from_=0, to=len(tsteps) - 1)
            self.time_slider.set(0)
            self.time_idx_label.config(text=f"1 / {len(tsteps)}")
        else:
            self.time_slider.configure(from_=0, to=0)
            self.time_slider.set(0)
            self.time_idx_label.config(text="0 / 0")

        # Draw preview
        self._refresh_preview_async()

    # ---- Apply settings ----
    def _apply_representation(self):
        if not self.session:
            return
        mode = self.repr_var.get()
        try:
            self.session.set_representation(mode)
            self._refresh_preview_async()
        except Exception as e:
            self._set_status(f"Failed to set representation: {e}", error=True)

    def _apply_coloring(self):
        if not self.session:
            return
        val = self.array_var.get()
        if ":" in val:
            assoc, name = val.split(":", 1)
        else:
            assoc, name = "POINTS", val
        try:
            self.session.set_coloring(assoc.strip(), name.strip())
            self._refresh_preview_async()
        except Exception as e:
            self._set_status(f"Failed to set coloring: {e}", error=True)

    def _on_time_slider(self, _evt=None):
        if not self.session:
            return
        idx = int(float(self.time_slider.get()))
        try:
            self.session.set_time_step_index(idx)
            total = int(self.time_slider.cget("to")) + 1
            self.time_idx_label.config(text=f"{idx+1} / {total}")
            self._refresh_preview_async()
        except Exception as e:
            self._set_status(f"Failed to set time: {e}", error=True)

    def _camera_reset(self):
        if self.session:
            self.session.camera_reset()
            self._refresh_preview_async()

    def _camera_azimuth(self, deg: float):
        if self.session:
            self.session.camera_azimuth(deg)
            self._refresh_preview_async()

    def _camera_elevation(self, deg: float):
        if self.session:
            self.session.camera_elevation(deg)
            self._refresh_preview_async()

    def _rescale_colors(self):
        if self.session:
            self.session.rescale_color_to_data()
            self._refresh_preview_async()

    # ---- Preview rendering ----
    def _refresh_preview_async(self):
        # Debounce rapid resize triggers by scheduling after idle
        self.after_idle(self._refresh_preview)

    def _refresh_preview(self):
        if not self.session or self.session.view is None:
            self._draw_info_on_canvas("No preview available. Load a supported dataset.")
            return
        # Compute snapshot size based on canvas size (cap to reasonable bounds)
        w = max(200, min(1920, self.canvas.winfo_width()))
        h = max(150, min(1200, self.canvas.winfo_height()))
        png = self.session.snapshot(w, h)
        if png is None or not png.exists():
            # Surface the subprocess stderr if available
            err = ""
            try:
                err_getter = getattr(self.session, "last_error", None)
                if callable(err_getter):
                    err = err_getter()
            except Exception:
                err = ""
            if err:
                self._set_status(err, error=True)
                self._draw_info_on_canvas("Failed to generate preview.\n\n" + err[:800])
            else:
                self._draw_info_on_canvas("Failed to generate preview.")
            return
        try:
            # Tk PhotoImage supports PNG in most modern builds
            self.preview_img = tk.PhotoImage(file=human_path(png))
            self.canvas.delete("all")
            self.canvas.create_image(w // 2, h // 2, image=self.preview_img)
        except Exception:
            # Fallback: show text
            self._draw_info_on_canvas(
                f"Preview saved to:\n{human_path(png)}\n\n(Your Tk build can't show PNG.)"
            )

    def _draw_info_on_canvas(self, text: str):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.canvas.create_text(
            w // 2,
            h // 2,
            text=text,
            fill="#ddd",
            font=("Arial", 12),
            anchor="c",
            width=int(w * 0.85),
            justify="center",
        )

    # ---- Export ----
    def _export_screenshot(self):
        if not self.session or self.session.view is None:
            messagebox.showwarning("Nothing to export", "Load a dataset first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Screenshot",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        w = max(640, self.canvas.winfo_width())
        h = max(480, self.canvas.winfo_height())
        ok, msg = self.session.export_screenshot(Path(path), w=w, h=h)
        self._set_status(msg, error=not ok)
        if ok:
            messagebox.showinfo("Screenshot saved", msg)
        else:
            messagebox.showerror("Export failed", msg)

    # ---- External ParaView GUI ----
    def _open_in_paraview_gui(self):
        exe = self.paraview_exe_var.get().strip()
        data = self.data_path_var.get().strip()
        if not data:
            messagebox.showwarning("No dataset", "Choose a dataset first.")
            return
        if not exe:
            # Try to find on PATH
            exe = which_exe("paraview") or which_exe("paraview.exe") or ""
            if not exe and platform.system().lower().startswith("win"):
                exe = DEFAULT_PARAVIEW_WIN
        if not Path(exe).exists():
            messagebox.showerror(
                "ParaView not found", f"ParaView executable not found:\n{exe}"
            )
            return
        try:
            subprocess.Popen(
                [exe, data], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self._set_status(f"Opening in ParaView GUI: {data}")
        except Exception as e:
            self._set_status(f"Failed to launch ParaView GUI: {e}", error=True)
            messagebox.showerror(
                "Launch failed", f"Could not open ParaView GUI:\n\n{e}"
            )

    # ---- Misc ----
    def _show_supported(self):
        msg = (
            "Supported (best effort):\n"
            " - XDMF/XMF (.xdmf, .xmf)\n"
            " - PVD (.pvd)\n"
            " - VTK Legacy (.vtk)\n"
            " - VTU/PVTU (.vtu, .pvtu)\n"
            " - ADIOS2 (.bp) — requires ADIOS2VTXReader plugin/build\n"
            " - HDF5 (.h5, .hdf5) via companion .xdmf\n"
            " - NPZ (.npz) metadata only (no rendering)\n\n"
            "Tip: For HDF5-based results, ensure an XDMF sidecar that describes the topology/geometry."
        )
        messagebox.showinfo("Supported Formats", msg)

    def _about(self):
        tip = "If ParaView Python is not importable, run this with 'pvpython' from your ParaView install."
        pv_ok = "Yes" if PARAVIEW_AVAILABLE else f"No ({PARAVIEW_IMPORT_ERROR})"
        mode = (
            "In-process (paraview.simple)"
            if PARAVIEW_AVAILABLE
            else "Subprocess via pvpython"
        )
        msg = (
            f"{APP_NAME}\n\n"
            "A simple, focused previewer for simulation outputs using the ParaView Python API.\n\n"
            f"Mode: {mode}\n"
            f"ParaView Python importable: {'Yes' if PARAVIEW_AVAILABLE else f'No ({PARAVIEW_IMPORT_ERROR})'}\n"
            "If ParaView Python is not importable, run this with 'pvpython' or let me call pvpython as a subprocess.\n"
        )
        messagebox.showinfo("About", msg)

    def _set_status(self, text: str, error: bool = False):
        self.status_var.set(text)
        self.update_idletasks()

    # ---- PV executable resolution ----
    def _resolve_pvpython_exe(self) -> Optional[str]:
        # Prefer inferring from the GUI exe entry (handles quotes and spaces)
        gui_path = (self.paraview_exe_var.get() or "").strip()
        cand = infer_pvpython_from_gui_path(gui_path)
        if cand:
            return cand
        # Fallback to PATH
        return which_exe("pvpython") or which_exe("pvpython.exe")


def _diagnose_environment(initial_path: Optional[Path]) -> None:
    """
    Print a canonical debug report to stdout/stderr to help diagnose issues like
    'No module named paraview' or reader problems.
    """
    sep = "-" * 72
    print(sep)
    print(f"{APP_NAME} — Verbose Environment Diagnostic")
    print(sep)
    try:
        print(f"Python executable : {sys.executable}")
        print(f"Python version    : {sys.version.split()[0]}")
        print(f"Platform          : {platform.platform()}")
        print(f"Working directory : {os.getcwd()}")
        print(f"argv              : {sys.argv}")
        print("")
        # Dataset / reader guess
        if initial_path is not None:
            print(f"Dataset path      : {human_path(initial_path)}")
            print(f"Exists            : {initial_path.exists()}")
            print(f"Suffix            : {initial_path.suffix.lower()}")
            try:
                # Reuse the same guessing logic as the session
                sess = PVSession(initial_path)
                rname, realp = sess._guess_reader()
                print(f"Guessed reader    : {rname}")
                if realp is not None:
                    print(f"Reader file path  : {human_path(realp)}")
                print(f"ADIOS2VTX avail   : {ADIOS2VTXReader is not None}")
            except Exception as e:
                print(f"Reader guess err  : {e}")
        else:
            print("Dataset path      : (none provided)")
        print("")
        # ParaView Python importability
        print("ParaView Python availability")
        print(f" - PARAVIEW_AVAILABLE : {PARAVIEW_AVAILABLE}")
        if PARAVIEW_AVAILABLE:
            try:
                import paraview

                print(
                    f" - paraview module   : {getattr(paraview, '__file__', 'unknown')}"
                )
            except Exception:
                print(
                    " - paraview module   : import succeeded earlier but __file__ unavailable"
                )
        else:
            print(f" - Import error      : {PARAVIEW_IMPORT_ERROR}")
            print(" - Traceback:")
            print(PARAVIEW_IMPORT_TRACEBACK)
        print("")
        # Executable discovery
        pvpython = which_exe("pvpython") or which_exe("pvpython.exe")
        paraview_exe = which_exe("paraview") or which_exe("paraview.exe")
        print("Executable discovery")
        print(f" - pvpython on PATH  : {pvpython or '(not found)'}")
        print(f" - paraview on PATH  : {paraview_exe or '(not found)'}")
        if platform.system().lower().startswith("win"):
            print(f" - Default Windows ParaView path check:")
            print(
                f"   {DEFAULT_PARAVIEW_WIN} -> {'exists' if Path(DEFAULT_PARAVIEW_WIN).exists() else 'missing'}"
            )
        print("")
        # Environment variables of interest
        print("Environment variables")
        env_keys = ["PYTHONPATH", "PATH", "PV_PLUGIN_PATH"]
        for k in env_keys:
            v = os.environ.get(k, "")
            print(f" - {k}:")
            if not v:
                print("   (empty)")
            else:
                sepchar = ";" if platform.system().lower().startswith("win") else ":"
                for item in v.split(sepchar):
                    print(f"   {item}")
        print("")
        # sys.path dump
        print("sys.path entries")
        for p in sys.path:
            print(f" - {p}")
        print("")

        # End of diagnostic
        print(sep)
        print("")

    except Exception as e:
        print(f"[diagnostic error] {e}", file=sys.stderr)


# -------------------------
# Entry point
# -------------------------


def main():
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} — quick ParaView-based viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Relative/absolute path to a dataset (.xdmf/.xmf, .pvd, .pvtu/.vtu, .vtk, .bp, .h5 with companion .xdmf, .npz)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print a detailed environment/reader diagnostic before launching the GUI",
    )
    args = parser.parse_args()

    initial_path = Path(args.path) if args.path else None

    # Optional pre-GUI diagnostics
    if args.verbose:
        _diagnose_environment(initial_path)

    app = PVQuickLookApp(initial_path=initial_path if initial_path else None)
    app.mainloop()
    if app.session is not None:
        app.session.cleanup()


if __name__ == "__main__":
    main()
