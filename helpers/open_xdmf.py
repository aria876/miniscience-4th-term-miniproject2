# open_xdmf.py
#
# Usage:
#   "C:\Program Files\LLNL\VisIt3.4.2\visit.exe" -cli -s helpers\open_xdmf.py -- path\to\file.xdmf
#
# This script opens an XDMF file in VisIt CLI mode with verbose logging.

import sys
import os
import time


def log(msg):
    """Print with timestamp and flush immediately."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=== VisIt XDMF Loader Script Started ===")
    log(f"Current working directory: {os.getcwd()}")
    log(f"Raw sys.argv: {sys.argv}")

    if len(sys.argv) < 2:
        log("ERROR: No .xdmf file path provided.")
        log("Usage: visit.exe -cli -s open_xdmf.py -- path\\to\\file.xdmf")
        sys.exit(1)

    # Resolve to absolute path
    xdmf_path = os.path.abspath(sys.argv[1])
    log(f"Resolved XDMF path: {xdmf_path}")

    if not os.path.isfile(xdmf_path):
        log(f"ERROR: File not found: {xdmf_path}")
        sys.exit(1)

    log("Attempting to open database...")
    ok = OpenDatabase(xdmf_path)
    log(f"OpenDatabase() returned: {ok}")

    if not ok:
        log("ERROR: VisIt failed to open the file. Check format/reader support.")
        sys.exit(1)

    log("Retrieving variable names...")
    vars = GetVariableNames()
    log(f"Variables found: {vars}")

    if vars:
        var_to_plot = vars[0]
        log(f"Adding Pseudocolor plot for variable: {var_to_plot}")
        AddPlot("Pseudocolor", var_to_plot)
        DrawPlots()
        log("Plot drawn successfully.")
    else:
        log("WARNING: No variables found in the file.")

    log("=== Script Finished ===")


if __name__ == "__main__":
    main()
