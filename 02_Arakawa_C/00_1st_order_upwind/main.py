"""
********************************************************************************
2D lid-driven cavity flow simulation with Finite Difference Method

space:
    convection:        1st order upwind
    diffusion:         2nd order central
    pressure gradient: 2nd order central

time:
    projection method (Chorin 1968)

grid:
    staggered grid (Arakawa C-type grid, Arakawa&Lamb 1977)

    + --↑-- + --↑-- + --↑-- + --↑-- + --↑-- +
    |       |       |       |       |       |
    →   *   →   *   →   *   →   *   →   *   →
    |       |       |       |       |       |
    + --↑-- + --↑-- + --↑-- + --↑-- + --↑-- +
    |       |       |       |       |       |
    →   *   →   *   →   *   →   *   →   *   →
    |       |       |       |       |       |
    + --↑-- + --↑-- + --↑-- + --↑-- + --↑-- +
    |       |       |       |       |       |
    →   *   →   *   →   *   →   *   →   *   →
    |       |       |       |       |       |
    + --↑-- + --↑-- + --↑-- + --↑-- + --↑-- +

    →: u-velocity
    ↑: v-velocity
    *: pressure
********************************************************************************
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
from reference import *

################################################################################

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dx",
        type=float,
        default=5e-3,
        help="grid spacing"
    )
    parser.add_argument(
        "-r",
        "--Re",
        type=float,
        default=1000.,
        help="Reynolds number"
    )
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        default=200.,
        help="maximum simulation time"
    )
    args = parser.parse_args()
    return args

################################################################################

def main(args):
    # arguments
    args = args
    dx = args.dx
    Re = args.Re

    # visualization setting
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["legend.framealpha"] = 1.
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["savefig.dpi"] = 300

    # domain
    Lx, Ly = 1., 1.
    dx, dy = dx, dx
    x = np.arange(0., Lx + dx, dx)
    y = np.arange(0., Ly + dy, dy)
    Nx, Ny = len(x), len(y)
    X, Y = np.meshgrid(x, y)

    # timestep
    U = 1.           # characteristic velocity
    h = min(dx, dy)  # characteristic length (discretization parameter)
    k = 1. / Re      # diffusion rate

    dt1 = h / U                # CFL
    dt2 = 1. / 2. * h**2 / k   # diffusion
    dt = min(dt1, dt2)         # critical timestep
    dt *= .2                   # safety

    # coefficient for numerical diffusion
    beta = 1.

    # variables
    u = np.zeros((Ny+1, Nx)) + 1e-3
    v = np.zeros((Ny, Nx+1)) + 1e-3
    p = np.zeros((Ny+1, Nx+1)) + 1e-3
    b = np.zeros((Ny+1, Nx+1)) + 1e-3

    u_tol = 1e-8         # convergence tolerance for velocity
    p_tol = 1e-5         # convergence tolerance for pressure
    it_max = int(1e4)     # max iteration

    # reference solutions
    ref_Ghia = Ghia(Re)
    ref_Erturk = Erturk(Re)

    df_Ghia = pd.DataFrame(ref_Ghia)
    df_Erturk = pd.DataFrame(ref_Erturk)

    # figure directory
    dir_res = f"Re{Re:.0f}"
    dir_fig_velocity_norm = os.path.join(dir_res, "velocity_norm")
    dir_fig_velocity_u = os.path.join(dir_res, "velocity_u")
    dir_fig_velocity_v = os.path.join(dir_res, "velocity_v")
    dir_fig_pressure = os.path.join(dir_res, "pressure")
    dir_fig_divergence = os.path.join(dir_res, "divergence")
    dir_fig_divergence_hat = os.path.join(dir_res, "divergence_hat")
    dir_fig_vorticity = os.path.join(dir_res, "vorticity")
    dir_fig_u = os.path.join(dir_res, "u")
    dir_fig_v = os.path.join(dir_res, "v")
    os.makedirs(dir_res, exist_ok=True)
    os.makedirs(dir_fig_velocity_norm, exist_ok=True)
    os.makedirs(dir_fig_pressure, exist_ok=True)
    os.makedirs(dir_fig_divergence, exist_ok=True)
    os.makedirs(dir_fig_divergence_hat, exist_ok=True)
    os.makedirs(dir_fig_vorticity, exist_ok=True)
    os.makedirs(dir_fig_u, exist_ok=True)
    os.makedirs(dir_fig_v, exist_ok=True)

    # main loop
    n = 0
    t = 0.
    u_res = 9999.
    while u_res > u_tol:
        # previous velocity
        u_old = u.copy()
        v_old = v.copy()

        # intermediate velocity
        u_hat = u.copy()
        v_hat = v.copy()

        # 1st order derivatives
        u_x = (u_old[1:-1, 2:] - u_old[1:-1, :-2]) / (2. * dx)
        u_y = (u_old[2:, 1:-1] - u_old[:-2, 1:-1]) / (2. * dy)
        v_x = (v_old[1:-1, 2:] - v_old[1:-1, :-2]) / (2. * dx)
        v_y = (v_old[2:, 1:-1] - v_old[:-2, 1:-1]) / (2. * dy)

        # 2nd order derivatives
        u_xx = (u_old[1:-1, 2:] - 2. * u_old[1:-1, 1:-1] + u_old[1:-1, :-2]) / dx**2
        u_yy = (u_old[2:, 1:-1] - 2. * u_old[1:-1, 1:-1] + u_old[:-2, 1:-1]) / dy**2
        v_xx = (v_old[1:-1, 2:] - 2. * v_old[1:-1, 1:-1] + v_old[1:-1, :-2]) / dx**2
        v_yy = (v_old[2:, 1:-1] - 2. * v_old[1:-1, 1:-1] + v_old[:-2, 1:-1]) / dy**2

        # convection
        # 1st order upwind scheme -> 2nd order central difference & 2nd order numerical diffusion
        # interpolate u, v to cell edge (used in v_u_y, u_v_x)
        u_itpl = 1. / 4. * (
            u_old[2:-1, 1:] + u_old[2:-1, :-1] \
            + u_old[1:-2, 1:] + u_old[1:-2, :-1]
        )   # used in u_v_x
        v_itpl = 1. / 4. * (
            v_old[1:, 2:-1] + v_old[1:, 1:-2] \
            + v_old[:-1, 2:-1] + v_old[:-1, 1:-2]
        )   # used in v_u_y
        # convection
        u_u_x = u_old[1:-1, 1:-1] * u_x
        v_u_y = v_itpl * u_y
        u_v_x = u_itpl * v_x
        v_v_y = v_old[1:-1, 1:-1] * v_y
        # numerical diffusion
        nd_u_u_x = np.abs(u_old[1:-1, 1:-1]) * dx / 2. * u_xx
        nd_v_u_y = np.abs(v_itpl) * dy / 2. * u_yy
        nd_u_v_x = np.abs(u_itpl) * dx / 2. * v_xx
        nd_v_v_y = np.abs(v_old[1:-1, 1:-1]) * dy / 2. * v_yy
        # convection (numerical diffusion is scaled by beta)
        conv_u = u_u_x - beta * nd_u_u_x + v_u_y - beta * nd_v_u_y
        conv_v = u_v_x - beta * nd_u_v_x + v_v_y - beta * nd_v_v_y

        # diffusion
        # laplacian operator
        lap_u = u_xx + u_yy
        lap_v = v_xx + v_yy
        # diffusion
        diff_u = k * lap_u
        diff_v = k * lap_v

        # intermediate velocity
        u_hat[1:-1, 1:-1] = u_old[1:-1, 1:-1] + dt * (- conv_u + diff_u)
        v_hat[1:-1, 1:-1] = v_old[1:-1, 1:-1] + dt * (- conv_v + diff_v)

        # source term for PPE (for Arakawa C-type grid)
        u_hat_x = (u_hat[1:-1, 1:] - u_hat[1:-1, :-1]) / dx
        v_hat_y = (v_hat[1:, 1:-1] - v_hat[:-1, 1:-1]) / dy
        div_u_hat = u_hat_x + v_hat_y
        b[1:-1, 1:-1] = div_u_hat / dt

        # solve PPE with point Jacobi method
        for it in range(0, it_max+1):
            # previous pressure
            p_old = p.copy()

            # point Jacobi method (for Arakawa B-type and C-type grid)
            p[1:-1, 1:-1] = 1. / (2. * (dx**2 + dy**2)) \
                            * (
                                - b[1:-1, 1:-1] * dx**2 * dy**2 \
                                + (p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy**2 \
                                + (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx**2
                            )

            # boundary condition (for Arakawa B-type and C-type grid)
            p[0,  :] = p[1,  :]   # South (be careful with meshgrid)
            p[-1, :] = p[-2, :]   # North
            p[:,  0] = p[:,  1]   # West
            p[:, -1] = p[:, -2]   # East
            p[1, 1] = 0.   # bottom left corner

            # converged?
            p_res = np.sqrt(np.sum((p - p_old)**2)) / np.sqrt(np.sum(p_old**2))
            if it % 1000 == 0:
                print(f"  >>> PPE -> it: {it}, p_res: {p_res:.6e}")
            if p_res < p_tol:
                print("   >>> PPE converged")
                break

        # pressure gradient (cell centre -> cell edge since used in velocity correction)
        p_x = (p[1:-1, 2:-1] - p[1:-1, 1:-2]) / dx
        p_y = (p[2:-1, 1:-1] - p[1:-2, 1:-1]) / dy

        # next-state velocity
        u[1:-1, 1:-1] = u_hat[1:-1, 1:-1] - dt * p_x
        v[1:-1, 1:-1] = v_hat[1:-1, 1:-1] - dt * p_y

        # boundary condition for u
        u[0, :]  = - u[1, :]        # South (no-slip Dirichlet)
        u[-1, :] = U                # North
        u[:, 0]  = - u[:, 1]        # West
        u[:, -1] = - u[:, -2]       # East

        # boundary condition for v
        v[0, :]  = - v[1, :]        # South (no-slip Dirichlet)
        v[-1, :] = - v[-2, :]       # North
        v[:, 0]  = - v[:, 1]        # West
        v[:, -1] = - v[:, -2]       # East

        # converged?
        n += 1
        t += dt
        u_res = np.sqrt(np.sum((u - u_old)**2)) / np.sqrt(np.sum(u_old**2))
        print(f"\n****************************************************************")
        print(f">>> main -> it: {n:d}, t: {t:.3f}, dt: {dt:.3e}, u_res: {u_res:.6e}")
        print(f"****************************************************************")
        if u_res < u_tol:
            print("   >>> main converged")
            break
        if t > args.time:
            print("   >>> maximum simulation time reached")
            break

        if n % 1000 == 0:
            # interpolation
            u_itpl = (u[1:, :] + u[:-1, :]) / 2.
            v_itpl = (v[:, 1:] + v[:, :-1]) / 2.

            plt.figure(figsize=(5, 4))
            vmin, vmax = 0., 1.+1e-6
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            vel_norm = np.sqrt(u_itpl**2 + v_itpl**2)
            plt.contourf(X, Y, vel_norm, cmap="turbo", levels=levels, extend="both")
            plt.colorbar(ticks=ticks, extend="both", label=r"$||\mathbf{u}||$")
            plt.xlim(0., Lx)
            plt.ylim(0., Ly)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.title(f"Velocity norm ($\mathrm{{Re}}={Re}, t={t:.2f}$)")
            plt.tight_layout()
            plt.savefig(os.path.join(dir_res, f"velocity_norm.png"), dpi=300)
            plt.savefig(os.path.join(dir_fig_velocity_norm, f"velocity_norm_n{n:d}.png"), dpi=300)
            plt.close()

            plt.figure(figsize=(5, 4))
            vmin, vmax = 0., 1.+1e-6
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            plt.streamplot(
                X, Y, u_itpl, u_itpl, color=vel_norm, cmap="turbo"
            )
            plt.colorbar(ticks=ticks, extend="both", label=r"$\psi$")
            plt.xlim(0., Lx)
            plt.ylim(0., Ly)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.title(f"Streamline, ($\mathrm{{Re}}={Re}, t={t:.2f}$)")
            plt.tight_layout()
            plt.savefig(os.path.join(dir_res, f"streamline.png"), dpi=300)
            plt.close()

            # # divergence & vorticity (for Arakawa A-type grid)
            # u_x = (u[1:-1, 2:] - u[1:-1, :-2]) / (2. * dx)
            # u_y = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2. * dy)
            # v_x = (v[1:-1, 2:] - v[1:-1, :-2]) / (2. * dx)
            # v_y = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2. * dy)
            # div_u = u_x + v_y
            # vor_u = v_x - u_y

            # # divergence & vorticity (for Arakawa B-type grid)
            # # interpolate u, v to cell edge, 
            # # then calculate divergence and vorticity
            # u_x = 1. / 2. * (
            #     (u[1:-2, 2:-1] - u[1:-2, 1:-2]) / dx \
            #     + (u[2:-1, 2:-1] - u[2:-1, 1:-2]) / dx
            # )
            # u_y = 1. / 2. * (
            #     (u[2:-1, 1:-2] - u[1:-2, 1:-2]) / dy \
            #     + (u[2:-1, 2:-1] - u[1:-2, 2:-1]) / dy
            # )
            # v_x = 1. / 2. * (
            #     (v[1:-2, 2:-1] - v[1:-2, 1:-2]) / dx \
            #     + (v[2:-1, 2:-1] - v[2:-1, 1:-2]) / dx
            # )
            # v_y = 1. / 2. * (
            #     (v[2:-1, 1:-2] - v[1:-2, 1:-2]) / dy \
            #     + (v[2:-1, 2:-1] - v[1:-2, 2:-1]) / dy
            # )
            # div_u = u_x + v_y
            # vor_u = v_x - u_y

            plt.figure(figsize=(5, 4))
            u_x = (u[1:-1, 1:] - u[1:-1, :-1]) / dx
            v_y = (v[1:, 1:-1] - v[:-1, 1:-1]) / dy
            div_u = u_x + v_y
            vmin, vmax = - .025, .025
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            plt.contourf(X[:-1, :-1], Y[:-1, :-1], div_u, cmap="coolwarm", levels=levels, extend="both")
            plt.colorbar(ticks=ticks, extend="both", label=r"$\nabla \cdot \mathbf{u}$")
            plt.xlim(0., Lx)
            plt.ylim(0., Ly)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.title(f"Divergence ($\mathrm{{Re}}={Re}, t={t:.2f}$)")
            plt.tight_layout()
            plt.savefig(os.path.join(dir_res, f"divergence.png"), dpi=300)
            plt.savefig(os.path.join(dir_fig_divergence, f"divergence_n{n:d}.png"), dpi=300)
            plt.close()

            # plot divergence of intermediate velocity u_hat
            plt.figure(figsize=(5, 4))
            vmin, vmax = - .025, .025
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            plt.contourf(X[:-1, :-1], Y[:-1, :-1], div_u_hat, cmap="coolwarm", levels=levels, extend="both")
            plt.colorbar(ticks=ticks, extend="both", label=r"$\nabla \cdot \mathbf{\hat{u}}$")
            plt.xlim(0., Lx)
            plt.ylim(0., Ly)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.title(f"Divergence of intermediate vel ($\mathrm{{Re}}={Re}, t={t:.2f}$)")
            plt.tight_layout()
            plt.savefig(os.path.join(dir_res, f"divergence_hat.png"), dpi=300)
            plt.savefig(os.path.join(dir_fig_divergence_hat, f"divergence_hat_n{n:d}.png"), dpi=300)
            plt.close()

            plt.figure(figsize=(5, 4))
            u_y = (u[1:, :] - u[:-1, :]) / dy
            v_x = (v[:, 1:] - v[:, :-1]) / dx
            vor_u = v_x - u_y
            vmin, vmax = -5., 5.
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            plt.contourf(X, Y, vor_u, cmap="turbo", levels=levels, extend="both")
            plt.colorbar(ticks=ticks, extend="both", label=r"$\omega$")
            plt.xlim(0., Lx)
            plt.ylim(0., Ly)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.title(f"Vorticity ($\mathrm{{Re}}={Re}, t={t:.2f}$)")
            plt.tight_layout()
            plt.savefig(os.path.join(dir_res, f"vorticity.png"), dpi=300)
            plt.savefig(os.path.join(dir_fig_vorticity, f"vorticity_n{n:d}.png"), dpi=300)
            plt.close()

            plt.figure(figsize=(5, 4))
            vmin, vmax = -.1, .1
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            plt.contourf(X, Y, p[:-1, :-1] - np.mean(p[:-1, :-1]), cmap="turbo", levels=levels, extend="both")
            plt.colorbar(ticks=ticks, extend="both", label=r"$p$")
            plt.xlim(0., Lx)
            plt.ylim(0., Ly)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.title(f"Pressure ($\mathrm{{Re}}={Re}, t={t:.2f}$)")
            plt.tight_layout()
            plt.savefig(os.path.join(dir_res, f"pressure.png"), dpi=300)
            plt.savefig(os.path.join(dir_fig_pressure, f"pressure_n{n:d}.png"), dpi=300)
            plt.close()

            # compare with reference solutions
            plt.figure(figsize=(4, 4))
            plt.scatter(df_Ghia["u"],   df_Ghia["y"],   alpha=.7, marker="1", label="Ghia et al. 1982")
            plt.scatter(df_Erturk["u"], df_Erturk["y"], alpha=.7, marker="2", label="Erturk et al. 2005")
            plt.plot(u[:-1, int(Ny/2)], y, alpha=.7, color="k", ls="--", label="Present")
            plt.legend()
            plt.xlabel(r"$u$")
            plt.ylabel(r"$y$")
            plt.title(f"Horizontal velocity along the geometric center")
            plt.tight_layout()
            plt.savefig(os.path.join(dir_res, f"u.png"), dpi=300)
            plt.savefig(os.path.join(dir_fig_u, f"u_n{n:d}.png"), dpi=300)
            plt.close()

            plt.figure(figsize=(4, 4))
            plt.scatter(df_Ghia["x"],   df_Ghia["v"],   alpha=.7, marker="1", label="Ghia et al. 1982")
            plt.scatter(df_Erturk["x"], df_Erturk["v"], alpha=.7, marker="2", label="Erturk et al. 2005")
            plt.plot(x, v[int(Nx/2), :-1], alpha=.7, color="k", ls="--", label="Present")
            plt.legend()
            plt.xlabel(r"$x$")
            plt.ylabel(r"$v$")
            plt.title(f"Vertical velocity along the geometric center")
            plt.tight_layout()
            plt.savefig(os.path.join(dir_res, f"v.png"), dpi=300)
            plt.savefig(os.path.join(dir_fig_v, f"v_n{n:d}.png"), dpi=300)
            plt.close()

    # save data
    dir_npz = os.path.join(dir_res, "npz")
    os.makedirs(dir_npz, exist_ok=True)
    np.savez(
        os.path.join(dir_npz, "results.npz"),
        x=x, y=y, X=X, Y=Y, u=u, v=v, p=p
    )

################################################################################

if __name__ == "__main__":
    args = arguments()
    main(args)


