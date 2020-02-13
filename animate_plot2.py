# import matplotlib.pyplot as plt
# import matplotlib.tri as tri
# import numpy as np
# import matplotlib.animation as animation
# # from IPython.display import HTML
# x = np.random.rand(27)
# y = np.random.rand(27)
#
# fig, ax = plt.subplots(figsize=(5,5))
#
# def update(num):
#     ax.cla()
#     ax.set_aspect('equal')
#     ax.set(xlim=(0,1),ylim=(0,1))
#     triang = tri.Triangulation(x[:3+num], y[:3+num])
#     ax.triplot(triang, 'go-', lw=1)
#     return fig,
#
# ani = animation.FuncAnimation(fig, update, 25,interval=200, blit=True)
# ani.save('triplot_anim25.mp4', writer="ffmpeg",dpi=100)
# # HTML(ani.to_html5_video())


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from matplotlib import cm
from matplotlib.animation import ArtistAnimation
from scipy.integrate import ode as ode


def animate_plot2(nt, d, e, s, p, t):
    # plt.rcParams['animation.html'] = 'html5'

    x = p[0, :]  # x-coordinates of nodes
    y = p[1, :]  # y-coordinates of nodes
    nnodes = len(p[0])
    time_intervals = np.asarray(list(map(int, np.linspace(0, nt - 1, nt))))
    if nt > 50:
        time_intervals = np.asarray(list(map(int, np.linspace(0, nt - 1, 50))))
    data = {
        0: {
            0: {
                "title": 'displacement_x',
                "value": d[:nnodes]
            },
            1: {
                "title": 'displacement_y',
                "value": d[nnodes:]
            }
        },
        1: {
            0: {
                "title": 'strain_x',
                "value": e[:nnodes]
            },
            1: {
                "title": 'strain_y',
                "value": e[nnodes:2 * nnodes]
            },
            2: {
                "title": 'strain_shear',
                "value": e[2 * nnodes:3 * nnodes]
            }
        },
        2: {
            0: {
                "title": 'stress_x',
                "value": s[:nnodes]
            },
            1: {
                "title": 'stress_y',
                "value": s[nnodes:2 * nnodes]
            },
            2: {
                "title": 'stress_shear',
                "value": s[2 * nnodes:3 * nnodes]
            }
        }
    }

    z = data[0][0]["value"]
    fig = plt.figure(figsize=(6.1, 5), facecolor='w')
    ims = []
    # levs = np.linspace(0,6,100)
    for i in range(nt):
        x = x + 1000 * data[0][0]['value'][:, i]
        y = y + 1000 * data[0][1]['value'][:, i]
        triang = mtri.Triangulation(x, y, t.transpose())
        im = plt.tricontourf(triang, z[:, i], 10, cmap='plasma')
        plt.triplot(triang, lw=0.2)
        ims.append(im.collections)
    cbar = plt.colorbar(im)
    # cbar.set_clim(0, 6)
    # cbar.set_ticks(np.linspace(0, 6, 7))
    cbar.set_label('Displacement magnitude [m]')
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.axes().set_aspect('equal')

    ani = ArtistAnimation(fig, ims, interval=200, repeat=True)
    ani.save('bs.gif', writer='imagemagick')


def animate_plot_alt(nt, d, e, s, p, t, et):
    x = p[0, :]  # x-coordinates of nodes
    y = p[1, :]  # y-coordinates of nodes
    nnodes = len(p[0])
    time_intervals = np.asarray(list(map(int, np.linspace(0, nt - 1, nt))))
    if nt > 50:
        time_intervals = np.asarray(list(map(int, np.linspace(0, nt - 1, 50))))

    data = {
        0: {
            0: {
                "title": 'displacement_x',
                "units": '[m]',
                "value": d[:nnodes]
            },
            1: {
                "title": 'displacement_y',
                "units": '[m]',
                "value": d[nnodes:]
            }
        },
        1: {
            0: {
                "title": 'strain_x',
                "units": '[-]',
                "value": e[:nnodes]
            },
            1: {
                "title": 'strain_y',
                "units": '[-]',
                "value": e[nnodes:2 * nnodes]
            },
            2: {
                "title": 'strain_shear',
                "units": '[-]',
                "value": e[2 * nnodes:3 * nnodes]
            }
        },
        2: {
            0: {
                "title": 'stress_x',
                "units": '[Pa]',
                "value": s[:nnodes]
            },
            1: {
                "title": 'stress_y',
                "units": '[Pa]',
                "value": s[nnodes:2 * nnodes]
            },
            2: {
                "title": 'stress_shear',
                "units": '[Pa]',
                "value": s[2 * nnodes:3 * nnodes]
            }
        }
    }

    for k in range(3):
        var = data[k]
        for j in range(len(var)):
            label = var[j]['title']
            units = var[j]['units']
            z = var[j]['value']

            img = []
            triang = mtri.Triangulation(x, y, t.transpose())
            fig, ax = plt.subplots()
            # ax.set_aspect('equal', 'box')
            # fig.tight_layout()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            min_z = np.min(z)
            max_z = np.max(z)

            for i in time_intervals:
                # ax.cla()
                plt.cla()
                ax.set_aspect('equal', 'box')
                ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))
                # fig.tight_layout()
                ax.set_title(label + ' in time step = ' + str(i))
                c = ax.tricontourf(triang, z[:, i], 10, cmap='plasma')
                c.set_clim(min_z, max_z)
                plt.colorbar(c, cax=cax)
                img.append(c.collections)

            name = label + '.gif'
            anim_img = animation.ArtistAnimation(fig, img, interval=300, blit=True)
            anim_img.save(name, writer='imagemagick', bitrate=300)
