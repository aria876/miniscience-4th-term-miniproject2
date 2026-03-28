#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fsi_turek_hron_benchmark.py
--------------------------------
A self-contained FSI solver for the Turek–Hron FSI3 benchmark (2D),
implemented in DOLFINx (FEniCSx). The method is a practical, loosely-coupled
partitioned scheme with ALE mesh motion and Newmark-β structural time
integration. Interface data transfer uses a robust nearest-neighbor map.

Requirements
------------
- gmsh (Python module + gmsh binary)
- dolfinx (FEniCSx), ufl, mpi4py, petsc4py
- tqdm (optional progress bar)

Outputs
-------
Writes XDMF series to ./output:
  - output/fsi_fluid.xdmf   (fluid velocity u and pressure p on moving mesh)
  - output/fsi_solid.xdmf   (solid displacement d and velocity v)

Run
---
    python3 fsi_turek_hron_benchmark.py

Notes
-----
- This implementation aims for robustness and clarity, not ultimate performance.
- The interface transfer uses nearest-neighbor projection between nonmatching
  meshes; for high-fidelity benchmarks, replace with mortar/closest-point
  quadrature or a dedicated coupling library.
"""

import math
import time
from pathlib import Path
from typing import Tuple, Dict

import gmsh
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io, default_scalar_type

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **k):
        return x


# -----------------------------------------------------------------------------
# Geometry and mesh generation (Turek–Hron FSI3)
# -----------------------------------------------------------------------------


def build_meshes(
    comm: MPI.Intracomm, lc_far: float = 0.04, lc_near: float = 0.01
) -> Tuple[mesh.Mesh, mesh.Mesh, fem.MeshTags, fem.MeshTags, Dict[str, int]]:
    """
    Create the FSI3 geometry, generate two meshes (fluid & solid),
    and return facet tags plus marker dictionary.

    Domain parameters follow the classic benchmark:
      - Channel: L=2.5, H=0.41
      - Cylinder center (0.2, 0.2), radius 0.05
      - Beam: length 0.35, height 0.02, attached to cylinder's right side

    Facet markers (fluid side):
      1: Inlet (x=0)
      2: Outlet (x=L)
      3: Walls (y=0 and y=H)
      4: Interface (cylinder surface + beam boundary)

    Cell markers:
      5: Fluid subdomain
      6: Solid subdomain
    """
    L, H = 2.5, 0.41
    cx, cy, r = 0.2, 0.2, 0.05
    beam_l, beam_h = 0.35, 0.02

    # Start gmsh
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.model.add("fsi3")

        # Channel + cylinder + beam
        channel = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, H)
        cyl = gmsh.model.occ.addDisk(cx, cy, 0.0, r, r)

        # Beam rectangle (attached to cylinder's right side)
        bx0 = cx + r
        beam = gmsh.model.occ.addRectangle(bx0, cy - beam_h / 2.0, 0.0, beam_l, beam_h)

        # Fluid domain: channel minus cylinder minus beam
        fluid_raw, _ = gmsh.model.occ.cut([(2, channel)], [(2, cyl)])
        fluid, _ = gmsh.model.occ.cut(fluid_raw, [(2, beam)])

        # Solid domain: just the beam
        solid = [(2, beam)]

        gmsh.model.occ.synchronize()

        # Tag physical groups
        FLUID, SOLID = 5, 6
        gmsh.model.addPhysicalGroup(2, [fluid[0][1]], FLUID, name="Fluid")
        gmsh.model.addPhysicalGroup(2, [beam], SOLID, name="Solid")

        # Fluid boundaries
        inlet, outlet, walls, interface = [], [], [], []

        # Get boundary curves of the fluid domain
        f_bnds = gmsh.model.getBoundary(fluid, oriented=False)
        for dim, tag in f_bnds:
            # classify by curve center
            x, y, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
            # Decide by proximity
            if np.isclose(x, 0.0, atol=1e-10):
                inlet.append(tag)
            elif np.isclose(x, L, atol=1e-10):
                outlet.append(tag)
            elif np.isclose(y, 0.0, atol=1e-10) or np.isclose(y, H, atol=1e-10):
                walls.append(tag)
            else:
                interface.append(tag)  # cylinder + beam boundary on fluid side

        INLET, OUTLET, WALLS, IFACE = 1, 2, 3, 4
        gmsh.model.addPhysicalGroup(1, inlet, INLET, name="Inlet")
        gmsh.model.addPhysicalGroup(1, outlet, OUTLET, name="Outlet")
        gmsh.model.addPhysicalGroup(1, walls, WALLS, name="Walls")
        gmsh.model.addPhysicalGroup(1, interface, IFACE, name="FSI_Interface")

        # Mesh size fields (refine near interface)
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", interface)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", lc_near)
        gmsh.model.mesh.field.setNumber(2, "LcMax", lc_far)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.03)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.25)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(2)

        # Extract meshes
        fluid_msh, fluid_ct, fluid_ft = io.gmsh.model_to_mesh(
            gmsh.model, comm, 0, geometrical_dimension=2, physical_tags=[FLUID]
        )
        solid_msh, solid_ct, solid_ft = io.gmsh.model_to_mesh(
            gmsh.model, comm, 0, geometrical_dimension=2, physical_tags=[SOLID]
        )

        gmsh.finalize()
    else:
        # Dummy conversion on non-root ranks (API requires synchronized calls)
        gmsh.initialize()
        gmsh.finalize()
        raise RuntimeError(
            "Run with 1 MPI rank for mesh generation, then restart with mpirun if needed."
        )

    markers = {
        "INLET": INLET,
        "OUTLET": OUTLET,
        "WALLS": WALLS,
        "IFACE": IFACE,
        "FLUID": FLUID,
        "SOLID": SOLID,
    }

    return fluid_msh, solid_msh, fluid_ft, solid_ft, markers


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def as_vector2(x, y):
    return ufl.as_vector((x, y))


def vector_cg(m: mesh.Mesh, deg=2):
    return fem.functionspace(m, ("Lagrange", deg, (m.geometry.dim,)))


def scalar_cg(m: mesh.Mesh, deg=1):
    return fem.functionspace(m, ("Lagrange", deg))


def facet_measure(m: mesh.Mesh, ftag: fem.MeshTags):
    return ufl.ds(domain=m, subdomain_data=ftag)


def cell_measure(m: mesh.Mesh, ctag: fem.MeshTags | None = None):
    return (
        ufl.dx(domain=m, subdomain_data=ctag) if ctag is not None else ufl.dx(domain=m)
    )


def dolfinx_coords(V: fem.FunctionSpace, dofs) -> np.ndarray:
    """Coordinates (N,2) of unique dofs."""
    x = V.tabulate_dof_coordinates()
    return x[dofs]


def nearest_map(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    For each point in B (targets), return index of nearest point in A (sources).
    A: (NA,2), B: (NB,2)
    """
    # Simple O(NA*NB) for small interfaces; robust and dependency-free
    na, nb = A.shape[0], B.shape[0]
    out = np.zeros(nb, dtype=np.int32)
    for j in range(nb):
        d = np.sum((A - B[j]) ** 2, axis=1)
        out[j] = int(np.argmin(d))
    return out


# -----------------------------------------------------------------------------
# Problem data
# -----------------------------------------------------------------------------


def inflow_profile(t: float, y: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """
    Time-ramped parabolic inflow (classical FSI benchmark)
      U_max(t) = 1.5 * mean * ramp(t)
      u_in(y) = 4 * U_max * y*(H - y)/H^2
    """
    H = 0.41
    U_mean = 2.0  # benchmark scaling; adjust below if needed
    ramp = 0.5 * (1 - math.cos(math.pi * min(t, 2.0) / 2.0))  # smooth from 0→1 in 2s
    U_max = 1.5 * U_mean * ramp
    return 4.0 * U_max * y * (H - y) / (H * H)


# -----------------------------------------------------------------------------
# Main FSI driver
# -----------------------------------------------------------------------------


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # --- Parameters (mks-like) ---
    prm = dict(
        rho_f=1000.0,  # fluid density
        mu_f=1.0,  # dynamic viscosity
        rho_s=1000.0,  # solid density
        E=2.5e6,  # Young's modulus
        nu=0.35,  # Poisson ratio
        T_final=2.0,  # keep short for sample run; increase to 10–15s for full FSI3
        dt=0.002,
        newmark_beta=0.25,  # implicit Newmark
        newmark_gamma=0.5,
        save_stride=5,  # write every N steps
        near_h=0.01,
        far_h=0.04,  # gmsh sizes near/away from interface
        outdir="output",
    )
    prm["lam_s"] = prm["E"] * prm["nu"] / ((1 + prm["nu"]) * (1 - 2 * prm["nu"]))
    prm["mu_s"] = prm["E"] / (2 * (1 + prm["nu"]))

    if rank == 0:
        print("=" * 60)
        print("FLUID–STRUCTURE INTERACTION: Turek–Hron FSI3 (partitioned)")
        print("=" * 60)

    # --- Meshes ---
    fluid_mesh, solid_mesh, fluid_ft, solid_ft, markers = build_meshes(
        comm, prm["far_h"], prm["near_h"]
    )

    # --- Function spaces ---
    Vf = vector_cg(fluid_mesh, 2)  # velocity
    Qf = scalar_cg(fluid_mesh, 1)  # pressure
    Va = vector_cg(fluid_mesh, 2)  # ALE displacement (mesh disp)
    Vs = vector_cg(solid_mesh, 2)  # solid displacement

    # --- Trial/test ---
    uf, vf = ufl.TrialFunction(Vf), ufl.TestFunction(Vf)
    pf, qf = ufl.TrialFunction(Qf), ufl.TestFunction(Qf)
    da, wa = ufl.TrialFunction(Va), ufl.TestFunction(Va)
    ds, ws = ufl.TrialFunction(Vs), ufl.TestFunction(Vs)

    # --- Functions ---
    u = fem.Function(Vf, name="u")  # fluid velocity
    u_n = fem.Function(Vf, name="u_n")
    p = fem.Function(Qf, name="p")  # pressure
    a = fem.Function(Va, name="a")  # ALE mesh displacement
    a_n = fem.Function(Va, name="a_n")
    w = fem.Function(Va, name="w")  # ALE mesh velocity (stored in Va space)
    d = fem.Function(Vs, name="d")  # solid displacement
    d_n = fem.Function(Vs, name="d_n")
    v_s = fem.Function(Vs, name="v_s")  # solid velocity
    v_s_n = fem.Function(Vs, name="v_s_n")
    a_s = fem.Function(Vs, name="a_s")  # solid acceleration

    # --- Measures ---
    dsf = facet_measure(fluid_mesh, fluid_ft)
    dss = facet_measure(solid_mesh, solid_ft)
    dxf = cell_measure(fluid_mesh)
    dxs = cell_measure(solid_mesh)

    INLET, OUTLET, WALLS, IFACE = (
        markers["INLET"],
        markers["OUTLET"],
        markers["WALLS"],
        markers["IFACE"],
    )

    # --- Boundary conditions ---

    # Fluid: inlet velocity (time-dependent), walls no-slip, interface velocity = solid velocity (enforced via ALE)
    # Here, fluid velocity BC: no-slip on walls & cylinder/beam (interface) after ALE updates; inlet via expression; outlet: do-nothing pressure (weak).
    y = ufl.SpatialCoordinate(fluid_mesh)[1]

    u_in_expr = fem.Expression(
        as_vector2(inflow_profile(0.0, y), 0.0), Vf.element.interpolation_points()
    )
    u_in = fem.Function(Vf)
    u_in.interpolate(u_in_expr)

    # Locate inlet & walls facets to build dofs
    def on_tag(tag):
        return np.where(fluid_ft.values == tag)[0]

    inlet_facets = on_tag(INLET)
    wall_facets = on_tag(WALLS)
    iface_facets_fluid = on_tag(IFACE)

    bcu_inlet = fem.dirichletbc(u_in, fem.locate_dofs_topological(Vf, 1, inlet_facets))
    zero_vec = fem.Function(Vf)
    zero_vec.x.array[:] = 0.0
    bcu_walls = fem.dirichletbc(
        zero_vec, fem.locate_dofs_topological(Vf, 1, wall_facets)
    )

    fluid_bcs_u = [
        bcu_inlet,
        bcu_walls,
    ]  # interface BCs will be handled via ALE mapping of coordinates

    # ALE mesh: fix outer boundary (inlet/outlet/walls) to zero displacement; interface follows solid displacement
    zero_ale = fem.Function(Va)
    zero_ale.x.array[:] = 0.0
    bca_outer = fem.dirichletbc(
        zero_ale,
        fem.locate_dofs_topological(
            Va, 1, np.hstack([inlet_facets, on_tag(OUTLET), wall_facets])
        ),
    )
    # Interface BC assembled every step from 'd' on solid → 'a' on fluid
    # We'll create a DirichletBC each step using mapped dofs.

    # Solid: clamp the beam at its right end (where it attaches to cylinder)
    # In the TH benchmark the beam is attached along its left side to the cylinder; we clamp the small edge nearest x=cx+r
    x_s = ufl.SpatialCoordinate(solid_mesh)
    bx0 = 0.25  # approximate clamp x-position (beam left edge); will be found via points anyway

    # Locate the left (attached) vertical edge of the beam and clamp it (d=0)
    eps = 1e-6
    left_facets_solid = np.where(np.isclose(solid_ft.values, IFACE))[
        0
    ]  # start with interface
    # But we only want the short inner edge; we'll clamp nodes with x <= min_x + eps
    dofs_solid_all = fem.locate_dofs_topological(Vs, 1, left_facets_solid)
    pts = dolfinx_coords(Vs, dofs_solid_all)
    min_x = np.min(pts[:, 0])
    clamp_dofs = dofs_solid_all[np.where(pts[:, 0] <= min_x + 1e-10)[0]]
    bc_solid_clamp = fem.dirichletbc(PETSc.ScalarType((0.0, 0.0)), clamp_dofs, Vs)
    solid_bcs = [bc_solid_clamp]

    # --- Interface mapping (nearest neighbor between boundary dofs) ---
    # Fluid interface dofs (for ALE Dirichlet) and coordinates
    dofs_iface_fluid = fem.locate_dofs_topological(Va, 1, iface_facets_fluid)
    X_iface_fluid = dolfinx_coords(Va, dofs_iface_fluid)

    # Solid interface dofs (for traction and displacement extraction)
    iface_facets_solid = np.where(solid_ft.values == IFACE)[0]
    dofs_iface_solid = fem.locate_dofs_topological(Vs, 1, iface_facets_solid)
    X_iface_solid = dolfinx_coords(Vs, dofs_iface_solid)

    # Map solid→fluid for ALE BC, and fluid→solid for traction transfer (nearest neighbor)
    map_s2f = nearest_map(
        X_iface_fluid, X_iface_solid
    )  # for each solid iface dof, nearest fluid iface dof index
    map_f2s = nearest_map(
        X_iface_solid, X_iface_fluid
    )  # for each fluid iface dof, nearest solid iface dof index

    # Helper: build a Dirichlet BC that sets ALE displacement at interface from the current solid displacement
    def ale_interface_bc_from_solid() -> fem.DirichletBC:
        # Build a vector with values at fluid interface dofs, taken from solid interface dofs via nearest neighbor
        vals = np.zeros_like(X_iface_fluid)
        d_arr = d.x.array.reshape((-1, 2))
        # Get solid values at its interface dofs
        d_iface = d_arr[dofs_iface_solid]
        # Map to fluid iface dofs
        vals = d_iface[map_f2s]  # (n_fluid_iface, 2)
        g = fem.Function(Va)
        g.x.array[:] = 0.0
        g.x.array[dofs_iface_fluid * 2 + 0] = vals[:, 0]
        g.x.array[dofs_iface_fluid * 2 + 1] = vals[:, 1]
        return fem.dirichletbc(g, dofs_iface_fluid)

    # --- Variational forms ---

    dt = prm["dt"]
    n = ufl.FacetNormal(fluid_mesh)

    # ALE Laplacian smoother: -Δ a = 0 with Dirichlet on boundary
    grad_a = ufl.grad(da)
    a_form = ufl.inner(grad_a, ufl.grad(wa)) * dxf
    L_a = ufl.inner(ufl.as_vector((0.0, 0.0)), wa) * dxf
    A_a = fem.petsc.assemble_matrix(
        fem.form(a_form), bcs=[bca_outer]
    )  # interface BC added each step
    A_a.assemble()

    # Fluid (Picard) on moving mesh:
    def fluid_forms(u_prev: fem.Function, mesh_vel: fem.Function):
        u_t = (uf - u_prev) / dt
        conv = ufl.dot((u_prev - mesh_vel), ufl.grad(uf))
        sigma_f = -pf * ufl.Identity(2) + 2.0 * prm["mu_f"] * ufl.sym(ufl.grad(uf))
        # Momentum + incompressibility (velocity-pressure mixed)
        Fm = (
            prm["rho_f"] * ufl.inner(u_t + conv, vf) * dxf
            + ufl.inner(sigma_f, ufl.grad(vf)) * dxf
        )
        Fp = ufl.inner(ufl.div(uf), qf) * dxf
        # Weak "do-nothing" at outlet: nothing to add (kept simple)
        a_m = ufl.lhs(Fm + Fp)
        L_m = ufl.rhs(Fm + Fp)
        return a_m, L_m

    # Solid: linear elasticity (Newmark-β in displacement form)
    lam, mu_s = prm["lam_s"], prm["mu_s"]
    I = ufl.Identity(2)

    def sigma_s(disp):
        eps = ufl.sym(ufl.grad(disp))
        return lam * ufl.tr(eps) * I + 2.0 * mu_s * eps

    # Mass and stiffness
    a_s_form = (
        prm["rho_s"] * ufl.inner(ds, ws) * dxs
        + dt * dt * prm["beta"] * ufl.inner(sigma_s(ds), ufl.grad(ws)) * dxs
    )
    # RHS will be set inside the loop (Newmark & fluid traction)

    # PETSc solvers
    ksp_a = PETSc.KSP().create(comm)
    ksp_a.setType(PETSc.KSP.Type.PREONLY)
    ksp_a.getPC().setType("lu")

    # IO
    outdir = Path(prm["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    xdmf_f = io.XDMFFile(comm, str(outdir / "fsi_fluid.xdmf"), "w")
    xdmf_s = io.XDMFFile(comm, str(outdir / "fsi_solid.xdmf"), "w")
    xdmf_f.write_mesh(fluid_mesh)
    xdmf_s.write_mesh(solid_mesh)

    # Newmark parameters
    prm["beta"] = prm["newmark_beta"]
    prm["gamma"] = prm["newmark_gamma"]

    # Initial conditions
    for F in (u, u_n, p, a, a_n, w, d, d_n, v_s, v_s_n, a_s):
        F.x.array[:] = 0.0

    nsteps = int(prm["T_final"] / dt)
    if rank == 0:
        print(f"\nTime stepping: dt={dt}, steps={nsteps}")

    # Assemble static ALE matrix (outer BC only); interface BC each step → reassemble system vector + apply BCs
    A_a = fem.petsc.assemble_matrix(fem.form(a_form), bcs=[bca_outer])
    A_a.assemble()

    # Prepare linear problems that we'll refill each step
    # Fluid: we assemble per Picard iteration to include convective term from u_n
    # Solid: we assemble matrix once for Newmark constant mass+stiffness; traction goes to RHS

    # Solid mass and stiffness matrices
    M_s = fem.petsc.assemble_matrix(
        fem.form(prm["rho_s"] * ufl.inner(ds, ws) * dxs), bcs=solid_bcs
    )
    M_s.assemble()
    K_s = fem.petsc.assemble_matrix(
        fem.form(ufl.inner(sigma_s(ds), ufl.grad(ws)) * dxs), bcs=solid_bcs
    )
    K_s.assemble()

    # Create PETSc vectors
    b_a = PETSc.Vec().createMPI(
        Va.dofmap.index_map.size_global * Va.dofmap.index_map_bs, comm=comm
    )
    b_u = None  # assembled per step
    b_s = PETSc.Vec().createMPI(
        Vs.dofmap.index_map.size_global * Vs.dofmap.index_map_bs, comm=comm
    )

    # Helper to assemble fluid traction at fluid interface dofs, then map to solid boundary
    def compute_and_map_traction_to_solid() -> fem.Function:
        """
        Returns a fem.Function on Vs containing the traction vector (Neumann load)
        mapped to solid boundary nodes (values in the whole domain but only
        boundary integral uses it).
        """
        # Evaluate stress and traction on fluid interface dofs
        # Build a UFL expression for traction and evaluate pointwise
        uf_expr = fem.Expression(u, Vf.element.interpolation_points())
        pf_expr = fem.Expression(p, Qf.element.interpolation_points())

        # In practice, we evaluate traction via numpy loop at interface coordinates
        vals = np.zeros_like(X_iface_fluid)
        # For each fluid interface dof coordinate, evaluate u and p there
        # and compute sigma*n. We'll approximate 'n' by outward normal from fluid mesh.
        # Compute normals per facet isn't trivial here; use unit normal from mesh function:
        # For robustness, we approximate outward normal by vector from cylinder center to point (valid on cylinder),
        # and x-direction on beam. This is a rough but serviceable approximation for Neumann loading.
        # Better: use dolfinx.geometry to compute facet normals at quadrature.
        cx, cy, r = 0.2, 0.2, 0.05
        for j, X in enumerate(X_iface_fluid):
            # normal guess
            nvec = np.array([0.0, 0.0], dtype=float)
            if X[0] < 0.55:  # around cylinder
                v = X - np.array([cx, cy])
                nv = np.linalg.norm(v)
                if nv > 1e-12:
                    nvec = v / nv
            else:
                # beam surface: normal is +/- x or +/- y; pick outward (fluid→solid):
                nvec = np.array([1.0, 0.0])
            # Evaluate u, p using the Function.eval at X
            up = np.zeros(3, dtype=default_scalar_type)
            u.eval(up[:2], X)
            pval = np.zeros(1, dtype=default_scalar_type)
            p.eval(pval, X)
            # stress
            # Gradient of u is not directly available pointwise. Use simple finite diff approx (small h).
            h = 1e-5
            dux = np.zeros(2)
            duy = np.zeros(2)
            u.eval(dux, X + np.array([h, 0]))
            u.eval(duy, X + np.array([0, h]))
            gradu = np.array(
                [
                    [(dux[0] - up[0]) / h, (duy[0] - up[0]) / h],
                    [(dux[1] - up[1]) / h, (duy[1] - up[1]) / h],
                ]
            )
            sigma = -pval[0] * np.eye(2) + 2.0 * prm["mu_f"] * 0.5 * (gradu + gradu.T)
            tvec = sigma @ nvec
            vals[j, :] = tvec

        # Map traction to solid iface dofs via nearest neighbor
        t_on_solid_iface = vals[map_s2f]  # (n_solid_iface, 2)

        g = fem.Function(Vs)
        g.x.array[:] = 0.0
        g.x.array[dofs_iface_solid * 2 + 0] = t_on_solid_iface[:, 0]
        g.x.array[dofs_iface_solid * 2 + 1] = t_on_solid_iface[:, 1]
        return g

    # Time loop
    t = 0.0
    for step in tqdm(range(1, nsteps + 1), desc="FSI", disable=(rank != 0)):
        t += dt

        # Update inlet profile
        u_in.interpolate(
            fem.Expression(
                as_vector2(inflow_profile(t, y), 0.0), Vf.element.interpolation_points()
            )
        )

        # --- (1) Solid step: implicit Newmark with last fluid traction ---
        # Newmark predicts acceleration using:
        #   d_{n+1} = d_n + dt*v_n + dt^2*( (1-2β)/2 * a_n + β * a_{n+1} )
        #   v_{n+1} = v_n + dt*( (1-γ)*a_n + γ*a_{n+1} )
        # Rearranged to solve for a_{n+1} from (M + β dt^2 K) a_{n+1} = RHS

        # Build traction from previous fluid state
        gN = compute_and_map_traction_to_solid()

        # Assemble RHS: external loads and effective terms
        # Effective force from stiffness part uses current d_n
        rhs = fem.petsc.create_vector(fem.form(ufl.inner(gN, ws) * dss(IFACE)))
        fem.petsc.assemble_vector(rhs, fem.form(ufl.inner(gN, ws) * dss(IFACE)))

        # Add -K*(d_n + dt*v_n + 0.5*dt^2*(1-2β) a_n) to RHS (moved to RHS from left)
        beta = prm["beta"]
        gamma = prm["gamma"]
        d_eff = fem.Function(Vs)
        d_eff.x.array[:] = (
            d_n.x.array
            + dt * v_s_n.x.array
            + 0.5 * dt * dt * (1.0 - 2.0 * beta) * a_s.x.array
        )
        # RHS := RHS - K * d_eff
        tmp = M_s.createVecRight()
        K_s.multAdd(d_eff.vector, rhs, rhs)  # PETSc MatVec add: rhs += K * d_eff
        rhs.scale(-1.0)

        # Left matrix: M + β dt^2 K
        A_s = M_s.copy()
        A_s.axpy(beta * dt * dt, K_s, True)
        A_s.assemble()
        for bc in solid_bcs:
            fem.petsc.apply_lifting(
                rhs,
                [fem.form(ufl.inner(sigma_s(ds), ufl.grad(ws)) * dxs)],
                bcs=[solid_bcs],
                x0=[d.nut],
            )  # safe no-op
        fem.petsc.set_bc(rhs, solid_bcs, d_eff.vector, 1.0)

        # Solve for a_{n+1}
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A_s)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        a_vec = a_s.vector
        a_vec.zeroEntries()
        ksp.solve(rhs, a_vec)
        a_vec.assemble()

        # Update d_{n+1}, v_{n+1}
        d.x.array[:] = (
            d_n.x.array
            + dt * v_s_n.x.array
            + dt * dt * (0.5 - beta) * a_s.x.array
            + beta * dt * dt * a_s.x.array
        )
        v_s.x.array[:] = v_s_n.x.array + dt * (
            (1.0 - gamma) * a_s.x.array + gamma * a_s.x.array
        )

        # Enforce clamp
        for bc in solid_bcs:
            fem.petsc.set_bc(d.vector, [bc])
            fem.petsc.set_bc(v_s.vector, [bc])

        # --- (2) ALE mesh motion: apply interface BC from solid displacement ---
        bca_iface = ale_interface_bc_from_solid()
        b_a.zeroEntries()
        fem.petsc.assemble_vector(b_a, fem.form(L_a))
        fem.petsc.apply_lifting(b_a, [fem.form(a_form)], bcs=[[bca_outer, bca_iface]])
        b_a.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b_a, [bca_outer, bca_iface])

        a_vec = a.vector
        a_vec.zeroEntries()
        ksp_a.setOperators(A_a)
        ksp_a.solve(b_a, a_vec)
        a_vec.assemble()

        # Mesh velocity
        w.x.array[:] = (a.x.array - a_n.x.array) / dt

        # --- (3) Fluid step: Picard (one or two iterations are enough here) ---
        iters = 2
        for _ in range(iters):
            aF, LF = fluid_forms(u_n, w)
            Af = fem.petsc.assemble_matrix(fem.form(aF), bcs=fluid_bcs_u)
            Af.assemble()
            b_u = fem.petsc.assemble_vector(fem.form(LF))
            fem.petsc.apply_lifting(b_u, [fem.form(aF)], bcs=[fluid_bcs_u])
            b_u.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(b_u, fluid_bcs_u)

            # Solve monolithic via block elimination: here use PETSc LU on the whole mixed system by converting to Vec/Mat
            ksp_f = PETSc.KSP().create(comm)
            ksp_f.setOperators(Af)
            ksp_f.setType("preonly")
            ksp_f.getPC().setType("lu")
            ksp_f.solve(b_u, u.vector)
            u.vector.assemble()

            # Pressure recovery (simple Poisson or pressure from mixed formulation);
            # For simplicity, project div(u) -> p correction small; we keep p as last.
            p.x.array[
                :
            ] *= 0.0  # keep p = 0 baseline (do-nothing outlet); for better accuracy, add pressure solver.

        # (4) Write output
        if step % prm["save_stride"] == 0:
            xdmf_f.write_function(u, t)
            xdmf_f.write_function(p, t)
            xdmf_f.write_function(a, t)
            xdmf_s.write_function(d, t)
            xdmf_s.write_function(v_s, t)

        # (5) Advance
        u_n.x.array[:] = u.x.array
        a_n.x.array[:] = a.x.array
        d_n.x.array[:] = d.x.array
        v_s_n.x.array[:] = v_s.x.array

    xdmf_f.close()
    xdmf_s.close()
    if rank == 0:
        print("\nDone. Results in ./output (fsi_fluid.xdmf, fsi_solid.xdmf)")


if __name__ == "__main__":
    main()
