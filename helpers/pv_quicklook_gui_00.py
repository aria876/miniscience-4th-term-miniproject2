#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pv_quicklook_gui.py

Monolithic helper to visualize FEniCSx/VTK/XDMF/ADIOS2 results using ParaView's Python API,
without exposing the full ParaView GUI.

Usage:
    python pv_quicklook_gui.py <path-to-data-file-or-directory>

Features:
- Cross-platform Tk GUI to choose ParaView binary (defaults provided per OS).
- Uses ParaView's pvpython to run an embedded visualization script.
- Auto-detects reader via OpenDataFile; colors by the first available scalar field.
- If a vector field is present, optional Glyphs can be toggled in the launcher.
- Supports time series (PVD/XDMF/ADIOS2) with ParaView's standard time controls in the
  render window (no full ParaView UI).
- Optional headless screenshot mode.

Tested with ParaView 5.10–5.13 family. Your mileage may vary with older versions.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import textwrap
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# ---------------------------- Helper: default binary guesses ----------------------------

def guess_paraview_binary() -> str:
    """
    Return a best-guess path to the ParaView GUI binary (paraview),
    from which we'll infer pvpython in the same bin folder.
    """
    if sys.platform.startswith("win"):
        # Default requested by user
        candidate = r"C:\Program Files\ParaView 5.13.1\bin\paraview.exe"
        if os.path.isfile(candidate):
            return candidate
        # Try another common fallback
        for ver in ["5.13.1", "5.12.0", "5.11.1", "5.10.1"]:
            p = fr"C:\Program Files\ParaView {ver}\bin\paraview.exe"
            if os.path.isfile(p):
                return p
        # As last resort, try PATH
        pv = shutil.which("paraview.exe") or shutil.which("paraview")
        return pv or candidate
    elif sys.platform == "darwin":
        # Typical app bundle
        candidates = [
            "/Applications/ParaView-5.13.1.app/Contents/MacOS/paraview",
            "/Applications/ParaView-5.12.0.app/Contents/MacOS/paraview",
            "/Applications/ParaView-5.11.1.app/Contents/MacOS/paraview",
            "/Applications/ParaView.app/Contents/MacOS/paraview",
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return shutil.which("paraview") or candidates[0]
    else:
        # Linux / *nix
        for c in [
            "/usr/bin/paraview",
            "/usr/local/bin/paraview",
            "/opt/paraview/bin/paraview",
        ]:
            if os.path.isfile(c):
                return c
        return shutil.which("paraview") or "/usr/bin/paraview"


def derive_pvpython_from_paraview(paraview_path: str) -> str:
    """
    Given a path to paraview executable, try to find the sibling pvpython.
    """
    if not paraview_path:
        return ""
    bin_dir = os.path.dirname(paraview_path)
    exe = "pvpython.exe" if sys.platform.startswith("win") else "pvpython"
    candidate = os.path.join(bin_dir, exe)
    if os.path.isfile(candidate):
        return candidate
    # Some mac/Linux installs may place pvpython elsewhere in PATH
    pvpy = shutil.which(exe)
    return pvpy or candidate


# ---------------------------- Embedded pvpython script ----------------------------

def build_embedded_pv_script(data_path: str, do_glyphs: bool, headless_png: str | None) -> str:
    """
    Create and return a pvpython script as text that:
    - opens the file/dir,
    - sets up a default view,
    - colors by first scalar,
    - optionally adds glyphs for first vector field,
    - either Interact() (onscreen) or SaveScreenshot (headless).
    """
    # Escape backslashes for safe embedding
    data_path_escaped = data_path.replace("\\", "\\\\")
    png_path_escaped = (headless_png or "").replace("\\", "\\\\")
    glyphs_flag = "True" if do_glyphs else "False"
    headless_flag = "True" if bool(headless_png) else "False"

    return textwrap.dedent(f"""
        from paraview.simple import *
        import os, sys

        # Quiet down a bit
        try:
            Disconnect()
        except Exception:
            pass
        Connect()

        data_path = r"{data_path_escaped}"
        want_glyphs = {glyphs_flag}
        headless = {headless_flag}
        png_out = r"{png_path_escaped}"

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {{data_path}}")

        # Open the dataset with auto reader
        src = OpenDataFile(data_path)
        if src is None:
            raise RuntimeError("ParaView could not find a reader for: " + data_path)

        # Create a render view
        view = CreateView('RenderView')
        view.OrientationAxesVisibility = 1
        view.Background = [0.1, 0.1, 0.12]
        layout = CreateLayout('Layout #1')
        AssignViewToLayout(view=view, layout=layout, hint=0)

        # Show the dataset
        disp = Show(src, view)

        # Choose a coloring: prefer first point scalar, then cell scalar
        def pick_first(arr_info_list, assoc_label):
            for a in arr_info_list:
                # Prefer scalars
                if a.GetNumberOfComponents() == 1:
                    return a.Name, 'SCALARS', assoc_label
            # Else return first array if any
            if arr_info_list:
                a0 = arr_info_list[0]
                role = 'VECTORS' if a0.GetNumberOfComponents() in (2,3) else 'SCALARS'
                return a0.Name, role, assoc_label
            return None

        info = src.GetDataInformation()
        pinfo = info.GetPointDataInformation()
        cinfo = info.GetCellDataInformation()

        choice = (pick_first([pinfo.GetArrayInformation(i) for i in range(pinfo.GetNumberOfArrays())], 'POINTS')
                  or pick_first([cinfo.GetArrayInformation(i) for i in range(cinfo.GetNumberOfArrays())], 'CELLS'))

        # Default representation
        disp.Representation = 'Surface With Edges'

        # Color by chosen array if available
        vec_array_name = None
        if choice:
            arr_name, role, assoc = choice
            if assoc == 'POINTS':
                ColorBy(disp, ('POINTS', arr_name))
            else:
                ColorBy(disp, ('CELLS', arr_name))
            disp.RescaleTransferFunctionToDataRange(True, False)
            disp.SetScalarBarVisibility(view, True)

            # Track if this is vector-like
            # Retrieve components count via data info again
            if assoc == 'POINTS':
                arr_info = pinfo.GetArrayInformation(pinfo.GetArrayInformationIndex(arr_name))
            else:
                arr_info = cinfo.GetArrayInformation(cinfo.GetArrayInformationIndex(arr_name))
            if arr_info and arr_info.GetNumberOfComponents() in (2, 3):
                vec_array_name = arr_name

        # Optionally add glyphs for vectors (point-data vectors preferred)
        glyph = None
        if want_glyphs and vec_array_name is not None:
            glyph = Glyph(registrationName='Glyphs', Input=src, GlyphType='Arrow')
            glyph.Scalars = ['POINTS', '']
            glyph.Vectors = ['POINTS', vec_array_name]
            glyph.ScaleFactor = 0.1
            glyph.GlyphMode = 'Uniform Spatial Distribution'
            glyph.MaximumNumberOfSamplePoints = 5000
            gdisp = Show(glyph, view)
            gdisp.SetRepresentationType('Surface')
            gdisp.DiffuseColor = [1.0, 1.0, 1.0]
            Hide(src, view)  # focus on glyphs first
        else:
            glyph = None

        ResetCamera(view)
        view.Update()

        # If time available, show time annotation and enable time widgets
        # (User can use standard 't'/'b' keys or GUI toolbar in the render window)
        try:
            if hasattr(src, 'TimestepValues') and src.TimestepValues:
                # Add a pretty time annotation
                text = Text(registrationName='TimeText')
                text.Text = 't = {{:.6g}}'.format(src.TimestepValues[0])
                tdisp = Show(text, view)
                tdisp.Color = [1,1,1]
                # Add programmable filter to update text with time
                # (simple approach: not strictly necessary—ParaView UI shows time)
        except Exception:
            pass

        # Headless or interactive
        if headless:
            if not png_out:
                png_out = os.path.join(os.getcwd(), "paraview_screenshot.png")
            SaveScreenshot(png_out, view=view, FontScaling='Do not scale fonts', ImageResolution=[1920, 1080])
            print("Saved screenshot to:", png_out)
        else:
            Interact(view)
    """)


# ---------------------------- Tk GUI ----------------------------

class LauncherGUI(tk.Tk):
    def __init__(self, data_path: str):
        super().__init__()
        self.title("ParaView QuickLook Launcher")
        self.resizable(False, False)

        self.data_path = os.path.abspath(data_path)
        self.paraview_bin = tk.StringVar(value=guess_paraview_binary())
        self.headless = tk.BooleanVar(value=False)
        self.screenshot_path = tk.StringVar(value="")
        self.enable_glyphs = tk.BooleanVar(value=True)

        # --- Layout ---
        pad = {'padx': 8, 'pady': 6}

        # Data path (read-only)
        tk.Label(self, text="Data to open:").grid(row=0, column=0, sticky="w", **pad)
        tk.Label(self, text=self.data_path, fg="#2a6").grid(row=0, column=1, columnspan=2, sticky="w", **pad)

        # ParaView binary
        tk.Label(self, text="ParaView binary (paraview):").grid(row=1, column=0, sticky="w", **pad)
        self.entry_bin = tk.Entry(self, textvariable=self.paraview_bin, width=60)
        self.entry_bin.grid(row=1, column=1, sticky="we", **pad)
        tk.Button(self, text="Browse…", command=self.browse_paraview).grid(row=1, column=2, **pad)

        # Glyph toggle
        tk.Checkbutton(self, text="Add vector glyphs if a vector field exists",
                       variable=self.enable_glyphs).grid(row=2, column=0, columnspan=3, sticky="w", **pad)

        # Headless mode
        tk.Checkbutton(self, text="Headless (save screenshot instead of interactive window)",
                       variable=self.headless, command=self.toggle_headless).grid(row=3, column=0, columnspan=3, sticky="w", **pad)

        tk.Label(self, text="Screenshot path (PNG):").grid(row=4, column=0, sticky="w", **pad)
        self.entry_png = tk.Entry(self, textvariable=self.screenshot_path, width=60, state="disabled")
        self.entry_png.grid(row=4, column=1, sticky="we", **pad)
        self.btn_png = tk.Button(self, text="Choose…", command=self.browse_png, state="disabled")
        self.btn_png.grid(row=4, column=2, **pad)

        # Launch buttons
        tk.Button(self, text="Launch", command=self.launch).grid(row=5, column=0, **pad)
        tk.Button(self, text="Quit", command=self.destroy).grid(row=5, column=2, sticky="e", **pad)

        # Info
        info = ("Tip: This does not open the full ParaView GUI. It launches a light-weight render window\n"
                "via ParaView's Python (pvpython). Use mouse + keyboard in that window to explore; close it to return.")
        tk.Label(self, text=info, fg="#888").grid(row=6, column=0, columnspan=3, sticky="w", padx=8, pady=(0,10))

    def browse_paraview(self):
        initdir = os.path.dirname(self.paraview_bin.get() or "")
        filetypes = [("Executable", "*.exe" if sys.platform.startswith("win") else "*")]
        path = filedialog.askopenfilename(initialdir=initdir, title="Select ParaView binary (paraview)", filetypes=filetypes)
        if path:
            self.paraview_bin.set(path)

    def toggle_headless(self):
        if self.headless.get():
            self.entry_png.config(state="normal")
            self.btn_png.config(state="normal")
        else:
            self.entry_png.config(state="disabled")
            self.btn_png.config(state="disabled")

    def browse_png(self):
        initdir = os.getcwd()
        path = filedialog.asksaveasfilename(initialdir=initdir, title="Save screenshot as",
                                            defaultextension=".png", filetypes=[("PNG image", "*.png")])
        if path:
            self.screenshot_path.set(path)

    def launch(self):
        paraview_bin = self.paraview_bin.get().strip()
        if not paraview_bin or not os.path.isfile(paraview_bin):
            messagebox.showerror("ParaView binary not found",
                                 "Please point to your ParaView executable (paraview).")
            return

        pvpython = derive_pvpython_from_paraview(paraview_bin)
        if not pvpython or not os.path.isfile(pvpython):
            messagebox.showerror("pvpython not found",
                                 "Could not locate pvpython next to the ParaView binary.\n"
                                 "Please ensure ParaView is properly installed.")
            return

        # Prepare the pvpython script in a temp file
        try:
            script_text = build_embedded_pv_script(
                data_path=self.data_path,
                do_glyphs=self.enable_glyphs.get(),
                headless_png=self.screenshot_path.get().strip() if self.headless.get() else None
            )
            tmp_dir =
