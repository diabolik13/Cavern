import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as mtri
import meshio
import inspect

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import (AnchoredOffsetbox, DrawingArea, HPacker,
                                  TextArea)


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def write_results_gif(input, output, l, ext, exaggerate=False):
    """Saves results in separate files in *.gif format."""

    nt = input['number of time steps']
    p = input['points']
    t = input['elements']

    x = p[0, :]  # x-coordinates of nodes
    y = p[1, :]  # y-coordinates of nodes
    nnodes = len(p[0])

    def ex(i):
        if exaggerate:
            return np.log(i + 1) * 1e3
        else:
            return 2e3

    # time_intervals = np.asarray(list(map(int, np.linspace(0, nt - 1, nt))))
    # if nt > 50:
    #     time_intervals = np.asarray(list(map(int, np.linspace(0, nt - 1, 50))))

    data = {
        0: {
            0: {
                "title": 'displacement_x',
                "units": '[m]',
                "value": output['displacement'][:nnodes]
            },
            1: {
                "title": 'displacement_y',
                "units": '[m]',
                "value": output['displacement'][nnodes:]
            }
        },
        1: {
            0: {
                "title": 'strain_x',
                "units": '[-]',
                "value": output['strain'][:nnodes]
            },
            1: {
                "title": 'strain_y',
                "units": '[-]',
                "value": output['strain'][nnodes:2 * nnodes]
            },
            2: {
                "title": 'strain_shear',
                "units": '[-]',
                "value": output['strain'][2 * nnodes:3 * nnodes]
            }
        },
        2: {
            0: {
                "title": 'stress_x',
                "units": '[Pa]',
                "value": output['stress'][:nnodes]
            },
            1: {
                "title": 'stress_y',
                "units": '[Pa]',
                "value": output['stress'][nnodes:2 * nnodes]
            },
            2: {
                "title": 'stress_shear',
                "units": '[Pa]',
                "value": output['stress'][2 * nnodes:3 * nnodes]
            }
        },
        3: {
            0: {
                "title": 'von_mises_stress',
                "units": '[Pa]',
                "value": output['Von Mises stress'][:nnodes]
            }
        },
        4: {
            0: {
                "title": 'creep_forces_x',
                "units": '[N]',
                "value": output['creep forces'][:nnodes]
            },
            1: {
                "title": 'creep_forces_y',
                "units": '[N]',
                "value": output['creep forces'][nnodes:]
            }
        }
    }

    if nt > 1:
        iter = len(data)
    elif nt == 1:
        iter = len(data) - 1

    for k in range(iter):
        var = data[k]
        for j in range(len(var)):
            folder = './output/'
            label = var[j]['title']
            units = var[j]['units']
            z = var[j]['value']

            fig, ax = plt.subplots()
            # fig.tight_layout()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            def animate(i):
                ax.cla()
                plt.cla()
                # fig.tight_layout()
                ax.set_aspect('equal', 'box')
                ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))
                # xc = x + ex(i) * data[0][0]['value'][:, i]
                # yc = y + ex(i) * data[0][1]['value'][:, i]
                triang = mtri.Triangulation(x, y, t.transpose())
                c = ax.tricontourf(triang, z[:, i], 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                                   levels=np.linspace(np.min(z), np.max(z), l))
                ax.triplot(triang, color='white', lw=0.1)
                # ax.set_title(
                #     label + ', elapsed time ' + "{:10.2f}".format((output['elapsed time'][i] / 86400)) + ' days.'
                # )
                cbar = plt.colorbar(c, cax=cax, format='%.0e')
                cbar.set_label(label + ' magnitude ' + units)
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')

            anim = FuncAnimation(
                fig, animate, interval=100, frames=nt)
            anim.save(folder + label + ext, writer='imagemagick')
    print("Done writing results to *.gif files.")


def write_results_xdmf(input, output):
    """Saves results in one file in *.xdmf format for ParaView."""

    nt = input['number of time steps']
    m = input['mesh data']
    p = input['points']

    nnodes = len(p[0])

    if nt > 1:
        time = np.round((output['elapsed time'] / 86400), 2)
    elif nt == 1:
        time = 0

    data = {
        0: {
            0: {
                "title": 'displacement_x',
                "units": '[m]',
                "value": output['displacement'][:nnodes]
            },
            1: {
                "title": 'displacement_y',
                "units": '[m]',
                "value": output['displacement'][nnodes:]
            }
        },
        1: {
            0: {
                "title": 'strain_x',
                "units": '[-]',
                "value": output['strain'][:nnodes]
            },
            1: {
                "title": 'strain_y',
                "units": '[-]',
                "value": output['strain'][nnodes:2 * nnodes]
            },
            2: {
                "title": 'strain_shear',
                "units": '[-]',
                "value": output['strain'][2 * nnodes:3 * nnodes]
            }
        },
        2: {
            0: {
                "title": 'stress_x',
                "units": '[Pa]',
                "value": output['stress'][:nnodes]
            },
            1: {
                "title": 'stress_y',
                "units": '[Pa]',
                "value": output['stress'][nnodes:2 * nnodes]
            },
            2: {
                "title": 'stress_shear',
                "units": '[Pa]',
                "value": output['stress'][2 * nnodes:3 * nnodes]
            }
        },
        3: {
            0: {
                "title": 'creep_forces_x',
                "units": '[N]',
                "value": output['creep forces'][:nnodes]
            },
            1: {
                "title": 'creep_forces_y',
                "units": '[N]',
                "value": output['creep forces'][nnodes:]
            }
        },
        4: {
            0: {
                "title": 'Von_Mises_stress',
                "units": '[Pa]',
                "value": output['Von Mises stress']
            }
        }
    }

    with meshio.xdmf.TimeSeriesWriter('./output/output_data.xdmf') as writer:
        writer.write_points_cells(m.points, m.cells)
        for i in range(nt):
            if i > 0:  # remove the very first frame with linear elastic response only
                writer.write_data(time[i],
                                  point_data={
                                      data[0][0]['title'] + ', ' + data[0][0]['units']: data[0][0]['value'][:, i],
                                      data[0][1]['title'] + ', ' + data[0][1]['units']: data[0][1]['value'][:, i],
                                      data[1][0]['title'] + ', ' + data[1][0]['units']: data[1][0]['value'][:, i],
                                      data[1][1]['title'] + ', ' + data[1][1]['units']: data[1][1]['value'][:, i],
                                      data[1][2]['title'] + ', ' + data[1][2]['units']: data[1][2]['value'][:, i],
                                      data[2][0]['title'] + ', ' + data[2][0]['units']: data[2][0]['value'][:, i],
                                      data[2][1]['title'] + ', ' + data[2][1]['units']: data[2][1]['value'][:, i],
                                      data[2][2]['title'] + ', ' + data[2][2]['units']: data[2][2]['value'][:, i],
                                      data[3][0]['title'] + ', ' + data[3][0]['units']: data[3][0]['value'][:, i],
                                      data[3][1]['title'] + ', ' + data[3][1]['units']: data[3][1]['value'][:, i],
                                      data[4][0]['title'] + ', ' + data[4][0]['units']: data[4][0]['value'][:, i]
                                  })
    print("Done writing results to *.xdmf files.")

# def animate_parameter(nt, z, p, t, label):
#     x = p[0, :]  # x-coordinates of nodes
#     y = p[1, :]  # y-coordinates of nodes
#     th = np.asarray(list(map(int, np.linspace(0, nt - 1, nt))))
#     if nt > 50:
#         th = np.asarray(list(map(int, np.linspace(0, nt - 1, 50))))
#
#     img = []
#     triang = mtri.Triangulation(x, y, t.transpose())
#     fig, ax = plt.subplots()
#     ax.set_aspect('equal', 'box')
#     fig.tight_layout()
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     min_z = np.min(z)
#     max_z = np.max(z)
#
#     for i in th:
#         ax.set_title(label + ' in time step = ' + str(i))
#         c = ax.tricontourf(triang, z[:, i], 10, cmap='plasma')
#         c.set_clim(min_z, max_z)
#         plt.colorbar(c, cax=cax)
#         img.append(c.collections)
#
#     name = label + '.gif'
#     anim_img = animation.ArtistAnimation(fig, img, interval=300, blit=True)
#     anim_img.save(name, writer='imagemagick', bitrate=300)
