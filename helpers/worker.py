#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
worker.py — pvpython-side renderer for pv_quicklook_gui.py

This script is executed by ParaView's pvpython.exe. It loads a dataset,
optionally probes arrays/timesteps (JSON), or renders a PNG snapshot.

Example:
  pvpython worker.py --data path\to\file.xdmf --out_png frame.png --width 1280 --height 720
"""

import argparse
import json
import sys
from pathlib import Path

try:
    # Import paraview.simple in the pvpython interpreter.
    from paraview.simple import (
        GetActiveViewOrCreate,
        GetAnimationScene,
        GetTimeKeeper,
        XDMFReader,
        PVDReader,
        LegacyVTKReader,
        XMLUnstructuredGridReader,
        XMLPartitionedUnstructuredGridReader,
        OpenDataFile,
        Show,
        Render,
        ColorBy,
        GetDisplayProperties,
        SaveScreenshot,
        ResetCamera,
    )

    # Optional reader (depends on build/plugins)
    try:
        from paraview.simple import ADIOS2VTXReader as _ADIOS2VTXReader
    except Exception:
        _ADIOS2VTXReader = None
except Exception as e:
    print(f"[pvpython] import error: {e}", file=sys.stderr)
    sys.exit(3)


def human(p: str) -> str:
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(p)


def safe_save_screenshot(path: str, view, w: int, h: int, quality: int) -> None:
    """
    Call SaveScreenshot with only properties supported by the current ParaView build.
    - PNG: no Quality; some builds accept CompressionLevel, others don't.
    - JPEG: Quality is valid (0–100).
    If a kwargs set fails, we back off to a simpler call.
    """
    from pathlib import Path as _P

    ext = _P(path).suffix.lower()
    # Start with universally accepted args
    kwargs = {"ImageResolution": [int(w), int(h)]}

    if ext in (".jpg", ".jpeg"):
        # Try JPEG quality if supported
        kwargs["Quality"] = int(max(0, min(100, quality)))
    elif ext == ".png":
        # Some builds support CompressionLevel; harmless to drop if not
        kwargs["CompressionLevel"] = 0

    try:
        SaveScreenshot(path, view, **kwargs)
        return
    except Exception:
        # Remove optional keys and retry
        kwargs.pop("Quality", None)
        kwargs.pop("CompressionLevel", None)
        try:
            SaveScreenshot(path, view, **kwargs)
            return
        except Exception:
            # Final fallback: rely on defaults
            SaveScreenshot(path, view)


def guess_reader(path: Path):
    """
    Use ParaView's OpenDataFile to select the correct reader.
    This avoids touching reader-specific properties that vary across versions
    and have been observed to crash on Windows in some builds.
    """
    s = path.suffix.lower()
    # For raw HDF5, prefer a companion .xdmf if present
    if s in (".h5", ".hdf5"):
        x = path.with_suffix(".xdmf")
        if x.exists():
            return OpenDataFile(human(str(x)))
        # Fall through: OpenDataFile may still find a reader if available
    try:
        return OpenDataFile(human(str(path)))
    except Exception:
        return None


def collect_arrays(src):
    out = []
    try:
        info = src.GetProperty("PointArrayStatus")
        if info is not None and hasattr(info, "GetData"):
            d = info.GetData()
            for i in range(0, len(d), 2):
                out.append(("POINTS", d[i]))
    except Exception:
        pass
    try:
        info = src.GetProperty("CellArrayStatus")
        if info is not None and hasattr(info, "GetData"):
            d = info.GetData()
            for i in range(0, len(d), 2):
                out.append(("CELLS", d[i]))
    except Exception:
        pass
    # dedupe
    seen = set()
    uniq = []
    for a in out:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
    return uniq


def main():
    ap = argparse.ArgumentParser(description="pvpython worker for pv_quicklook_gui.py")
    ap.add_argument("--data", required=True, help="Path to dataset")
    ap.add_argument(
        "--probe_json", help="If set, write arrays/timesteps JSON here and exit"
    )
    ap.add_argument("--out_png", help="If set, write a rendered PNG here")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--repr", default="Surface")
    ap.add_argument("--color_assoc", default="")
    ap.add_argument("--color_array", default="")
    ap.add_argument("--time_index", type=int, default=-1)
    ap.add_argument("--rescale_colors", action="store_true")
    ap.add_argument("--cam_reset", action="store_true")
    ap.add_argument("--cam_azimuth", type=float, default=0.0)
    ap.add_argument("--cam_elevation", type=float, default=0.0)
    args = ap.parse_args()

    path = Path(args.data)
    src = guess_reader(path)
    if src is None:
        print(f"[pvpython] unsupported or unreadable file: {path}", file=sys.stderr)
        sys.exit(2)

    view = GetActiveViewOrCreate("RenderView")
    # Prefer offscreen rendering when running under pvpython (Windows headless etc.)
    try:
        pv_set(view, "UseOffscreenRendering", 1)
    except Exception:
        pass
    disp = Show(src, view)

    anim = GetAnimationScene()
    anim.UpdateAnimationUsingDataTimeSteps()
    tk = GetTimeKeeper()

    if args.probe_json:
        tsteps = list(getattr(tk, "TimestepValues", []))
        arrays = collect_arrays(src)
        out = {
            "reader": (
                src.SMProxy.GetXMLLabel() if hasattr(src, "SMProxy") else "unknown"
            ),
            "arrays": arrays,
            "timesteps": tsteps,
        }
        Path(args.probe_json).write_text(json.dumps(out), encoding="utf-8")
        return

    # Representation
    try:
        GetDisplayProperties(src, view=view).SetRepresentationType(args.repr)
    except Exception:
        pass

    # Coloring
    if args.color_array:
        assoc = "POINTS" if args.color_assoc.upper().startswith("POINT") else "CELLS"
        try:
            ColorBy(disp, (assoc, args.color_array))
            if args.rescale_colors and hasattr(
                disp, "RescaleTransferFunctionToDataRange"
            ):
                disp.RescaleTransferFunctionToDataRange(True, False)
        except Exception:
            ColorBy(disp, None)

    # Time
    tsteps = list(getattr(tk, "TimestepValues", []))
    if tsteps and args.time_index >= 0:
        idx = max(0, min(args.time_index, len(tsteps) - 1))
        try:
            tk.Time = tsteps[idx]
        except Exception:
            try:
                anim.AnimationTime = tsteps[idx]
            except Exception:
                pass

    # Camera
    if args.cam_reset:
        ResetCamera(view)
    try:
        cam = view.GetActiveCamera()
        if args.cam_azimuth:
            cam.Azimuth(args.cam_azimuth)
        if args.cam_elevation:
            cam.Elevation(args.cam_elevation)
    except Exception:
        try:
            if args.cam_azimuth:
                view.Azimuth(args.cam_azimuth)
            if args.cam_elevation:
                view.Elevation(args.cam_elevation)
        except Exception:
            pass

    Render(view)

    if args.out_png:
        safe_save_screenshot(
            human(args.out_png),
            view,
            w=args.width,
            h=args.height,
            quality=args.quality,
        )


if __name__ == "__main__":
    main()
