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
    staggered grid (Arakawa B-type grid, Arakawa&Lamb 1977)
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


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dx", type=float, default=5e-3, help="grid spacing")
parser.add_argument("-r", "--Re", type=float, default=1000., help="Reynolds number")
parser.add_argument("-t", "--time", type=float, default=200., help="maximum simulation time")
parser.add_argument("-u", "--u_tol", type=float, default=1e-8, help="convergence tolerance for velocity")
parser.add_argument("-p", "--p_tol", type=float, default=1e-5, help="convergence tolerance for pressure")
parser.add_argument("-i", "--it_max", type=int, default=int(1e4), help="maximum iteration for PPE")
args = parser.parse_args()


def plot_setting():
    # visualization setting
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["legend.framealpha"] = 1.
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["savefig.dpi"] = 300


def main():
    # plot setting
    plot_setting()

    # arguments
    dx = args.dx
    Re = args.Re

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
    u = np.zeros(shape=(Ny, Nx)) + 1e-3
    v = np.zeros(shape=(Ny, Nx)) + 1e-3
    p = np.zeros(shape=(Ny+1, Nx+1)) + 1e-3
    b = np.zeros(shape=(Ny+1, Nx+1)) + 1e-3

    u_tol = args.u_tol    # convergence tolerance for velocity
    p_tol = args.p_tol    # convergence tolerance for pressure
    it_max = args.it_max  # max iteration

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
    os.makedirs(dir_fig_velocity_u, exist_ok=True)
    os.makedirs(dir_fig_velocity_v, exist_ok=True)
    os.makedirs(dir_fig_divergence_hat, exist_ok=True)
    os.makedirs(dir_fig_pressure, exist_ok=True)
    os.makedirs(dir_fig_divergence, exist_ok=True)
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
        # 1st order upwind -> 2nd order central difference + 2nd order numerical diffusion
        u_u_x = u_old[1:-1, 1:-1] * u_x - beta * abs(u_old[1:-1, 1:-1]) * dx / 2. * u_xx
        v_u_y = v_old[1:-1, 1:-1] * u_y - beta * abs(v_old[1:-1, 1:-1]) * dy / 2. * u_yy
        u_v_x = u_old[1:-1, 1:-1] * v_x - beta * abs(u_old[1:-1, 1:-1]) * dx / 2. * v_xx
        v_v_y = v_old[1:-1, 1:-1] * v_y - beta * abs(v_old[1:-1, 1:-1]) * dy / 2. * v_yy
        conv_u = u_u_x + v_u_y
        conv_v = u_v_x + v_v_y

        # diffusion
        lap_u = u_xx + u_yy
        lap_v = v_xx + v_yy
        diff_u = k * lap_u
        diff_v = k * lap_v

        # intermediate velocity
        u_hat[1:-1, 1:-1] = u_old[1:-1, 1:-1] + dt * (- conv_u + diff_u)
        v_hat[1:-1, 1:-1] = v_old[1:-1, 1:-1] + dt * (- conv_v + diff_v)

        # source term for PPE (for Arakawa B-type grid)
        # interpolate u_hat, v_hat to cell center
        u_hat_itpl = 1. / 2. * (u_hat[1:, :] + u_hat[:-1, :])
        v_hat_itpl = 1. / 2. * (v_hat[:, 1:] + v_hat[:, :-1])
        u_hat_x = (u_hat_itpl[:, 1:] - u_hat_itpl[:, :-1]) / dx
        v_hat_y = (v_hat_itpl[1:, :] - v_hat_itpl[:-1, :]) / dy
        div_u_hat = u_hat_x + v_hat_y
        b[1:-1, 1:-1] = div_u_hat / dt

        # # source term for PPE (for Arakawa B-type grid)
        # # interpolate u_hat, v_hat to cell edge
        # u_hat_x = 1. / 2. * (
        #     (u_hat[1:-2, 2:-1] - u_hat[1:-2, 1:-2]) / dx \
        #     + (u_hat[2:-1, 2:-1] - u_hat[2:-1, 1:-2]) / dx
        # )
        # v_hat_y = 1. / 2. * (
        #     (v_hat[2:-1, 1:-2] - v_hat[1:-2, 1:-2]) / dy \
        #     + (v_hat[2:-1, 2:-1] - v_hat[1:-2, 2:-1]) / dy
        # )
        # div_u_hat = u_hat_x + v_hat_y   # divergence mapped to cell center
        # b[1:-1, 1:-1] = div_u_hat / dt

        # solve PPE with point Jacobi method
        for it in range(0, it_max+1):
            # previous pressure
            p_old = p.copy()

            # # point Jacobi method (for Arakawa A-type grid)
            # p[2:-2, 2:-2] = 1. / (2. * (dx**2 + dy**2)) \
            #                 * (
            #                     - b[2:-2, 2:-2] * dx**2 * dy**2 \
            #                     + (p_old[2:-2, 3:-1] + p_old[2:-2, 1:-3]) * dy**2 \
            #                     + (p_old[3:-1, 2:-2] + p_old[1:-3, 2:-2]) * dx**2
            #                 )

            # point Jacobi method (for Arakawa B-type grid)
            p[1:-1, 1:-1] = 1. / (2. * (dx**2 + dy**2)) \
                            * (
                                - b[1:-1, 1:-1] * dx**2 * dy**2 \
                                + (p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy**2 \
                                + (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx**2
                            )

            # # boundary condition (for Arakawa A-type grid)
            # p[1,  :] = p[2,  :]   # South (be careful with meshgrid)
            # p[-2, :] = p[-3, :]   # North
            # p[:,  1] = p[:,  2]   # West
            # p[:, -2] = p[:, -3]   # East
            # # p[1, int(nx/2)] = 0.   # bottom center
            # p[1, 1] = 0.   # bottom left corner

            # boundary condition (for Arakawa B-type grid)
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
                print("   >>> ppe converged")
                break

        # pressure gradient
        # interpolate p to cell edge
        p_x = 1. / 2. * (
            (p[2:-1, 2:-1] - p[2:-1, 1:-2]) / dx \
            + (p[1:-2, 2:-1] - p[1:-2, 1:-2]) / dx
        )
        p_y = 1. / 2. * (
            (p[2:-1, 1:-2] - p[1:-2, 1:-2]) / dy \
            + (p[2:-1, 2:-1] - p[1:-2, 2:-1]) / dy
        )

        # velocity correction
        u[1:-1, 1:-1] = u_hat[1:-1, 1:-1] - dt * p_x
        v[1:-1, 1:-1] = v_hat[1:-1, 1:-1] - dt * p_y

        # boundary condition
        u[0,  :], v[0,  :] = 0., 0.   # South (be careful with meshgrid)
        u[-1, :], v[-1, :] = 1., 0.   # North
        u[:,  0], v[:,  0] = 0., 0.   # West
        u[:, -1], v[:, -1] = 0., 0.   # East

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
            print("   >>> taking too long, terminating now...")
            break

        if n % 1000 == 0:
            plt.figure(figsize=(5, 4))
            vmin, vmax = 0., 1.+1e-6
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            vel_norm = np.sqrt(u**2 + v**2)
            plt.contourf(X, Y, vel_norm, cmap="turbo", levels=levels, extend="both")
            plt.colorbar(ticks=ticks, extend="both", label=r"$||\mathbf{u}||$")
            plt.xlim(0., Lx)
            plt.ylim(0., Ly)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.title(f"Velocity norm ($\mathrm{{Re}}={Re}, t={t:.2f}, \mathrm{{res}}={u_res:.2e}$)")
            plt.tight_layout()
            plt.savefig(os.path.join(dir_res, f"velocity_norm.png"), dpi=300)
            plt.savefig(os.path.join(dir_fig_velocity_norm, f"velocity_norm_n{n:d}.png"), dpi=300)
            plt.close()

            plt.figure(figsize=(5, 4))
            vmin, vmax = 0., 1.+1e-6
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            plt.contourf(X, Y, u, cmap="turbo", levels=levels, extend="both")
            plt.colorbar(ticks=ticks, extend="both", label=r"$u$")
            plt.xlim(0., Lx)
            plt.ylim(0., Ly)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.title(f"Horizontal velocity ($\mathrm{{Re}}={Re}, t={t:.2f}$)")
            plt.tight_layout()
            plt.savefig(os.path.join(dir_res, f"velocity_u.png"), dpi=300)
            plt.savefig(os.path.join(dir_fig_velocity_u, f"velocity_u_n{n:d}.png"), dpi=300)
            plt.close()

            plt.figure(figsize=(5, 4))
            vmin, vmax = -.5, .5+1e-6
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            plt.contourf(X, Y, v, cmap="turbo", levels=levels, extend="both")
            plt.colorbar(ticks=ticks, extend="both", label=r"$u$")
            plt.xlim(0., Lx)
            plt.ylim(0., Ly)
            plt.xlabel(r"$x$")
            plt.ylabel(r"$y$")
            plt.title(f"Vertical velocity ($\mathrm{{Re}}={Re}, t={t:.2f}$)")
            plt.tight_layout()
            plt.savefig(os.path.join(dir_res, f"velocity_v.png"), dpi=300)
            plt.savefig(os.path.join(dir_fig_velocity_v, f"velocity_v_n{n:d}.png"), dpi=300)
            plt.close()

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
            plt.title(f"Divergence of intermediate velocity ($\mathrm{{Re}}={Re}, t={t:.2f}$)")
            plt.tight_layout()
            plt.savefig(os.path.join(dir_res, f"divergence_hat.png"), dpi=300)
            plt.savefig(os.path.join(dir_fig_divergence_hat, f"divergence_hat_n{n:d}.png"), dpi=300)
            plt.close()

            plt.figure(figsize=(5, 4))
            vmin, vmax = 0., 1.+1e-6
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            plt.streamplot(
                X, Y, u, v, color=vel_norm, cmap="turbo"
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

            # interpolate u, v to cell edge
            u_itpl = 1. / 2. * (u[1:, :] + u[:-1, :])   # interpolation along y-axis
            v_itpl = 1. / 2. * (v[:, 1:] + v[:, :-1])   # interpolation along x-axis

            # divergence
            u_x = (u_itpl[:, 1:] - u_itpl[:, :-1]) / dx
            v_y = (v_itpl[1:, :] - v_itpl[:-1, :]) / dy
            div_u = u_x + v_y

            plt.figure(figsize=(5, 4))
            vmin, vmax = - .025, .025
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            plt.contourf(X[:-1, :-1] + dx / 2., Y[:-1, :-1] + dy / 2., div_u, cmap="coolwarm", levels=levels, extend="both")
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

            # vorticity
            u_y = (u[1:, :-1] - u[:-1, :-1]) / dy
            v_x = (v[:-1, 1:] - v[:-1, :-1]) / dx
            vor_u = v_x - u_y

            plt.figure(figsize=(5, 4))
            vmin, vmax = -5., 5.
            levels = np.linspace(vmin, vmax, 32)
            ticks = np.linspace(vmin, vmax, 5)
            plt.contourf(X[:-1, :-1] + dx / 2., Y[:-1, :-1] + dy / 2., vor_u, cmap="coolwarm", levels=levels, extend="both")
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
            plt.plot(u[:, int(Ny/2)], y, alpha=.7, color="k", ls="--", label="Present")
            plt.legend(loc="lower right")
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
            plt.plot(x, v[int(Nx/2), :], alpha=.7, color="k", ls="--", label="Present")
            plt.legend(loc="upper right")
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


if __name__ == "__main__":
    main()


