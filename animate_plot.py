import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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


def write_results(input, output, l, ext, exaggerate=False):
    """Saves results in separate files in *.gif and *.png formats."""

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

            # fig, ax = plt.subplots()
            fig = plt.figure(figsize=(11, 9))
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            # fig.tight_layout()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            def animate(i):
                ax.cla()
                plt.cla()
                # fig.tight_layout()
                ax.set_aspect('equal', 'box')
                ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))
                xc = x + ex(i) * data[0][0]['value'][:, i]
                yc = y + ex(i) * data[0][1]['value'][:, i]
                triang = mtri.Triangulation(xc, yc, t.transpose())
                c = ax.tricontourf(triang, z[:, i], l, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                                   levels=np.linspace(np.min(z), np.max(z), l))
                ax.locator_params(axis='both', nbins=3)
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=16)
                ax.triplot(triang, color='white', lw=0.1)
                ax.set_title(
                    label + ', elapsed time ' + "{:10.2f}".format((output['elapsed time'][i] / 86400)) + ' days.\n',
                    fontsize=30)
                cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
                cbar.set_label(label + ' magnitude ' + units, fontsize=30)
                cbar.ax.tick_params(labelsize=30)
                # cbar.ax.ticklabel_format(useMathText=True)
                ax.set_xlabel('x [m]', fontsize=30)
                ax.set_ylabel('y [m]', fontsize=30)
                ax.ticklabel_format(useMathText=True)

            if nt < 20:
                anim = FuncAnimation(
                    fig, animate, interval=100, frames=nt)
                anim.save(folder + label + ext, writer='imagemagick')
            else:
                anim = FuncAnimation(
                    fig, animate, interval=100, frames=20)
                anim.save(folder + label + ext, writer='imagemagick')
    print('Done writing results to *' + ext + ' files.')


def write_results_xdmf(input, output):
    """Saves results in one file in *.xdmf format for ParaView."""

    nt = input['number of time steps']
    m = input['mesh data']
    p = input['points']

    nnodes = len(p[0])

    if nt > 1:
        time = np.round((output['elapsed time'] / 86400), 2)
    elif nt == 1:
        time = np.array([0])

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
            # if i > 0:  # remove the very first frame with linear elastic response only
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


def write_results_xdmf2(nt, mesh, output, folder):
    """Saves results in one file in *.xdmf format for ParaView."""
    from pathlib import Path

    nnodes = mesh.nnodes()

    if nt > 1:
        time = np.round((output['elapsed time'] / 86400), 2)
    elif nt == 1:
        time = np.array([0])

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

    Path('./output/' + folder).mkdir(parents=True, exist_ok=True)
    with meshio.xdmf.TimeSeriesWriter('./output/' + folder + '/output_data.xdmf') as writer:
        writer.write_points_cells(mesh.meshdata().points, mesh.meshdata().cells)
        for i in range(nt):
            # if i > 0:  # remove the very first frame with linear elastic response only
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
    print("Done writing results to *.xdmf files in the ./" + folder + " folder.")


def save_plot(input, output, node):
    """Saves results for one particular point."""

    from matplotlib.ticker import ScalarFormatter

    class ScalarFormatterForceFormat(ScalarFormatter):
        def _set_format(self):  # Override function that finds format to use.
            self.format = "%1.1f"  # Give format here

    nt = input['number of time steps']
    p = input['points']
    t = input['elements']

    x = p[0, :]  # x-coordinates of nodes
    y = p[1, :]  # y-coordinates of nodes
    nnodes = len(p[0])

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
            folder = './output/A/'
            label = var[j]['title']
            units = var[j]['units']
            z = var[j]['value']

            # fig, ax = plt.subplots(constrained_layout=True)
            # fig.tight_layout()
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_axes([0.2, 0.15, 0.7, 0.7])
            # ax.cla()
            # plt.cla()
            # ax.set_aspect('equal', 'box')
            ax.set(xlim=(np.min(output['elapsed time']) / 86400,
                         np.max(output['elapsed time']) / 86400))
            # ylim=(np.min(z[node]), np.max(z[node])))

            ax.plot(output['elapsed time'] / 86400, z[node], 'ro-')
            # yfmt = ScalarFormatterForceFormat()
            # yfmt.set_powerlimits((0, 0))
            # ax.yaxis.set_major_formatter(yfmt)
            # ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.2e}'))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.ticklabel_format(useMathText=True)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            ax.locator_params(axis='both', nbins=5)
            ax.grid(True)
            ax.set_xlabel('elapsed time, [days]', fontsize=16)
            ax.set_ylabel(label + ', ' + units, fontsize=16)
            plt.savefig(folder + label + '_A.png')
            plt.close('all')
            # plt.show()

    print('Done writing results to the output files.')


def save_plot2(input, output, node):
    """Saves results for one particular point."""

    from matplotlib.ticker import ScalarFormatter

    class ScalarFormatterForceFormat(ScalarFormatter):
        def _set_format(self):  # Override function that finds format to use.
            self.format = "%1.1f"  # Give format here

    nt = input['number of time steps']
    p = input['points']
    t = input['elements']

    x = p[0, :]  # x-coordinates of nodes
    y = p[1, :]  # y-coordinates of nodes
    nnodes = len(p[0])

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

            # fig, ax = plt.subplots(constrained_layout=True)
            # fig.tight_layout()
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
            ax.cla()
            plt.cla()
            # ax.set_aspect('equal', 'box')
            # ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))

            ax.plot(output['elapsed time'] / 86400, z[node], 'ro-')
            # yfmt = ScalarFormatterForceFormat()
            # yfmt.set_powerlimits((0, 0))
            # ax.yaxis.set_major_formatter(yfmt)
            # ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.2e}'))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.ticklabel_format(useMathText=True)
            ax.grid(True)
            ax.set_xlabel('elapsed time, [days]', fontsize=20)
            ax.set_ylabel(label + ', ' + units, fontsize=20)
            plt.savefig(folder + label + '_vs_disp.png')
            # plt.show()

    print('Done writing results to the output files.')


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

def write_results2(nt, p, t, output, l, ext, exaggerate=False):
    """
    Saves results in separate files in *.gif and *.png formats.
    """

    x, y = p
    nnodes = len(x)

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
            # },
            # 2: {
            #     0: {
            #         "title": 'stress_x',
            #         "units": '[Pa]',
            #         "value": output['stress'][:nnodes]
            #     },
            #     1: {
            #         "title": 'stress_y',
            #         "units": '[Pa]',
            #         "value": output['stress'][nnodes:2 * nnodes]
            #     },
            #     2: {
            #         "title": 'stress_shear',
            #         "units": '[Pa]',
            #         "value": output['stress'][2 * nnodes:3 * nnodes]
            #     }
            # },
            # 3: {
            #     0: {
            #         "title": 'von_mises_stress',
            #         "units": '[Pa]',
            #         "value": output['Von Mises stress'][:nnodes]
            #     }
            # },
            # 4: {
            #     0: {
            #         "title": 'creep_forces_x',
            #         "units": '[N]',
            #         "value": output['creep forces'][:nnodes]
            #     },
            #     1: {
            #         "title": 'creep_forces_y',
            #         "units": '[N]',
            #         "value": output['creep forces'][nnodes:]
            #     }
        }
    }

    if nt > 1:
        iter = len(data)
    elif nt == 1:
        iter = len(data) - 1

    for k in data.keys():
        var = data[k]
        for j in range(len(var)):
            folder = './output/'
            label = var[j]['title']
            units = var[j]['units']
            z = var[j]['value']

            # fig, ax = plt.subplots()
            fig = plt.figure(figsize=(11, 9))
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            # fig.tight_layout()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            def animate(i):
                ax.cla()
                plt.cla()
                # fig.tight_layout()
                ax.set_aspect('equal', 'box')
                ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))
                xc = x + ex(i) * data[0][0]['value'][:, i]
                yc = y + ex(i) * data[0][1]['value'][:, i]
                triang = mtri.Triangulation(xc, yc, t.transpose())
                c = ax.tricontourf(triang, z[:, i], l, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                                   levels=np.linspace(np.min(z), np.max(z), l))
                ax.locator_params(axis='both', nbins=3)
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=16)
                ax.triplot(triang, color='white', lw=0.1)
                ax.set_title(
                    label + ', elapsed time ' + "{:10.2f}".format((output['elapsed time'][i] / 86400)) + ' days.\n',
                    fontsize=30)
                cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
                cbar.set_label(label + ' magnitude ' + units, fontsize=30)
                cbar.ax.tick_params(labelsize=30)
                # cbar.ax.ticklabel_format(useMathText=True)
                ax.set_xlabel('x [m]', fontsize=30)
                ax.set_ylabel('y [m]', fontsize=30)
                ax.ticklabel_format(useMathText=True)

            if nt < 20:
                anim = FuncAnimation(
                    fig, animate, interval=100, frames=nt)
                anim.save(folder + label + ext, writer='imagemagick')
            else:
                anim = FuncAnimation(
                    fig, animate, interval=100, frames=20)
                anim.save(folder + label + ext, writer='imagemagick')
    print('Done writing results to *' + ext + ' files.')
