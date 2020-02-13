import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as mtri
import meshio

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import (AnchoredOffsetbox, DrawingArea, HPacker,
                                  TextArea)


def load_mesh(mesh_filename):
    m = meshio.read(mesh_filename)
    p = m.points.transpose()
    p = np.delete(p, 2, axis=0)
    t = m.cells["triangle"]
    return p, t


def animate_plot_tmp(nt, p, t):
    x = p[0, :]
    y = p[1, :]
    triang = mtri.Triangulation(x, y, t)
    nnodes = len(p[0])
    z = np.zeros((nnodes, nt))
    for j in range(nt):
        z[:, j] = j ** 2 * (np.sin(x * 10) + np.sin(y * 10))

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    def animate(i):
        ax.cla()
        plt.cla()
        ax.set_aspect('equal', 'box')
        ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))
        c = ax.tricontourf(triang, z[:, i], 10, cmap='plasma', vmin=-1, vmax=1)
        c.set_clim(np.min(z), np.max(z))
        ax.triplot(triang, color='white', lw=0.1)
        ax.set_title('test, ' + 'np.min(z)=' + str(np.min(z)) + ', np.max(z)=' + str(np.max(z)) + '.')
        cbar = plt.colorbar(c, cax=cax, format='%.0e')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

    anim = FuncAnimation(
        fig, animate, interval=600, frames=nt)
    anim.save('test.gif', writer='imagemagick')


def animate_plot2_tmp(nt, p, t):
    x = p[0, :]
    y = p[1, :]
    triang = mtri.Triangulation(x, y, t)
    nnodes = len(p[0])
    z = np.zeros((nnodes, nt))
    for j in range(nt):
        z[:, j] = 100 * (np.sin(x * 10) + np.sin(y * 10)) - j ** 2 * (np.sin(x * 10) + np.sin(y * 10))

    img = []
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    for i in range(nt):
        # ax.cla()
        # plt.cla()
        ax.set_aspect('equal', 'box')
        ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))
        fig.tight_layout()
        ax.set_title('time step = ' + str(i))
        c = ax.tricontourf(triang, z[:, i], 10, cmap='plasma')
        c.set_clim(np.min(z), np.max(z))
        ax.triplot(triang, color='white', lw=0.1)
        plt.colorbar(c, cax=cax)
        img.append(c.collections)

    name = 'test.gif'
    anim_img = animation.ArtistAnimation(fig, img, interval=300, blit=True)
    anim_img.save(name, writer='imagemagick', bitrate=300)


mesh_filename = 'new_cave.msh'
p, t = load_mesh(mesh_filename)

nt = 10
animate_plot(nt, p, t)
